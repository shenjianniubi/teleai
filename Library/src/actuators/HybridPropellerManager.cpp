/*
    This file is a part of Stonefish.

    Stonefish is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Stonefish is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

//
//  HybridPropellerManager.cpp
//  Stonefish
//
//  Copyright (c) 2024-2025 [Your Name]. All rights reserved.
//

#include "actuators/HybridPropellerManager.h"
#include "core/SimulationApp.h"
#include "core/SimulationManager.h"
#include "graphics/OpenGLContent.h"
#include <iostream>
#include <thread>
#include <future>
#include <vector>

namespace sf
{

HybridPropellerManager::HybridPropellerManager(
    std::string uniqueName,
    std::shared_ptr<SolidEntity> propeller,    // 螺旋桨实体
    Scalar diameter,                                                        // 直径
    Scalar thrustCoeffAir, Scalar torqueCoeffAir, Scalar maxRPMAir,         // 空气推力系数  空气力矩系数  空气最大转速
    std::shared_ptr<RotorDynamics> rotorDynamicsWater,                      // 水下电机动力学模型
    std::shared_ptr<ThrustModel> thrustModelWater,                          // 水下推力模型
    Scalar maxSetpointWater,                                                // 水下最大控制输入
    bool rightHand, bool inverted, bool normalizedSetpoint
)
    : LinkActuator(uniqueName)
{
    RH_ = rightHand; 
    inv_ = inverted;
    normalized_ = normalizedSetpoint;
    setpoint_ = thrust_ = torque_ = Scalar(1);
    solid_ = propeller; // 存储 SolidEntity
    faceMeshPrecomputed_ = false; // 初始化预计算标志

    // 创建水下 Thruster
    thruster_ = std::make_shared<Thruster>(
        uniqueName + "_Thruster", propeller, rotorDynamicsWater, 
        thrustModelWater, diameter, rightHand, maxSetpointWater, 
        inverted, normalizedSetpoint);
    thruster_->setRelativeActuatorFrame(o2a);

    // 创建空中 Propeller
    propeller_ = std::make_shared<Propeller>(uniqueName + "_Propeller", propeller, diameter, 
                                            thrustCoeffAir, torqueCoeffAir, maxRPMAir, 
                                            rightHand, inverted);
    propeller_->setRelativeActuatorFrame(o2a);

    // 确保螺旋桨有图形对象
    propeller->BuildGraphicalObject();

    // 显式初始化为水下
    transitionFactor_ = 1.0;
    
    // 预计算网格面积
    precomputeMeshAreas();
}

void HybridPropellerManager::precomputeMeshAreas()
{
    if (solid_ == nullptr)
        return;
        
    const Mesh* mesh = solid_->getPhysicsMesh();
    if (mesh == nullptr)
        return;
        
    triangleAreas_.clear();
    triangleAreas_.resize(mesh->faces.size()); // 预分配空间避免动态扩容
    totalSurfaceArea_ = Scalar(0);
    
    // 优化的并行计算：使用线程池和块处理
    const size_t numFaces = mesh->faces.size();
    const unsigned int numThreads = std::max(1u, std::thread::hardware_concurrency()/16);
    const size_t chunkSize = std::max(size_t(1), numFaces / (numThreads * 4)); // 每个线程处理多个块
    
    std::vector<std::future<std::pair<size_t, Scalar>>> futures;
    futures.reserve((numFaces + chunkSize - 1) / chunkSize);
    
    // 分块并行处理
    for (size_t startIdx = 0; startIdx < numFaces; startIdx += chunkSize)
    {
        size_t endIdx = std::min(startIdx + chunkSize, numFaces);
        
        futures.emplace_back(std::async(std::launch::async, 
            [this, mesh, startIdx, endIdx]() -> std::pair<size_t, Scalar>
            {
                Scalar localTotalArea = Scalar(0);
                
                // 内层循环：批量计算减少函数调用开销
                for (size_t i = startIdx; i < endIdx; ++i)
                {
                    // 获取三角形的三个顶点（局部坐标）- 使用缓存友好的访问模式
                    glm::vec3 p1gl = mesh->getVertexPos(i, 0);
                    glm::vec3 p2gl = mesh->getVertexPos(i, 1);
                    glm::vec3 p3gl = mesh->getVertexPos(i, 2);
                    
                    // 向量化计算三角形面积 = 0.5 * |cross(p2-p1, p3-p1)|
                    glm::vec3 edge1 = p2gl - p1gl;
                    glm::vec3 edge2 = p3gl - p1gl;
                    glm::vec3 crossProduct = glm::cross(edge1, edge2);
                    Scalar area = 0.5f * glm::length(crossProduct);
                    
                    triangleAreas_[i] = area;
                    localTotalArea += area;
                }
                
                return std::make_pair(endIdx - startIdx, localTotalArea);
            }));
    }
    
    // 收集结果
    for (auto& future : futures)
    {
        auto result = future.get();
        totalSurfaceArea_ += result.second;
    }
    
    faceMeshPrecomputed_ = true;
}

Scalar HybridPropellerManager::calculateTriangleSubmergedArea(
    const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3,
    GLfloat depth1, GLfloat depth2, GLfloat depth3,
    Scalar originalArea) const
{
    // 计算有多少个顶点在水下
    int submergedCount = 0;
    if (depth1 >= 0.f) submergedCount++;
    if (depth2 >= 0.f) submergedCount++;
    if (depth3 >= 0.f) submergedCount++;
    
    if (submergedCount == 0)
        return Scalar(0); // 完全在水面上
    if (submergedCount == 3)
        return originalArea; // 完全浸没
        
    // 部分浸没情况，使用面积比例计算
    if (submergedCount == 1)
    {
        // 一个顶点在水下，计算小三角形面积比例
        glm::vec3 submergedPoint, abovePoint1, abovePoint2;
        GLfloat submergedDepth, aboveDepth1, aboveDepth2;
        
        if (depth1 >= 0.f)
        {
            submergedPoint = p1; submergedDepth = depth1;
            abovePoint1 = p2; aboveDepth1 = depth2;
            abovePoint2 = p3; aboveDepth2 = depth3;
        }
        else if (depth2 >= 0.f)
        {
            submergedPoint = p2; submergedDepth = depth2;
            abovePoint1 = p1; aboveDepth1 = depth1;
            abovePoint2 = p3; aboveDepth2 = depth3;
        }
        else
        {
            submergedPoint = p3; submergedDepth = depth3;
            abovePoint1 = p1; aboveDepth1 = depth1;
            abovePoint2 = p2; aboveDepth2 = depth2;
        }
        
        // 计算水线交点
        Scalar t1 = submergedDepth / (submergedDepth + fabsf(aboveDepth1));
        Scalar t2 = submergedDepth / (submergedDepth + fabsf(aboveDepth2));
        
        glm::vec3 intersect1 = submergedPoint + t1 * (abovePoint1 - submergedPoint);
        glm::vec3 intersect2 = submergedPoint + t2 * (abovePoint2 - submergedPoint);
        
        // 计算浸没三角形面积
        glm::vec3 edge1 = intersect1 - submergedPoint;
        glm::vec3 edge2 = intersect2 - submergedPoint;
        Scalar submergedArea = 0.5f * glm::length(glm::cross(edge1, edge2));
        
        return submergedArea;
    }
    else // submergedCount == 2
    {
        // 两个顶点在水下，计算大四边形面积（原面积减去水上小三角形）
        glm::vec3 abovePoint, submergedPoint1, submergedPoint2;
        GLfloat aboveDepth, submergedDepth1, submergedDepth2;
        
        if (depth1 < 0.f)
        {
            abovePoint = p1; aboveDepth = depth1;
            submergedPoint1 = p2; submergedDepth1 = depth2;
            submergedPoint2 = p3; submergedDepth2 = depth3;
        }
        else if (depth2 < 0.f)
        {
            abovePoint = p2; aboveDepth = depth2;
            submergedPoint1 = p1; submergedDepth1 = depth1;
            submergedPoint2 = p3; submergedDepth2 = depth3;
        }
        else
        {
            abovePoint = p3; aboveDepth = depth3;
            submergedPoint1 = p1; submergedDepth1 = depth1;
            submergedPoint2 = p2; submergedDepth2 = depth2;
        }
        
        // 计算水线交点
        Scalar t1 = submergedDepth1 / (submergedDepth1 + fabsf(aboveDepth));
        Scalar t2 = submergedDepth2 / (submergedDepth2 + fabsf(aboveDepth));
        
        glm::vec3 intersect1 = submergedPoint1 + t1 * (abovePoint - submergedPoint1);
        glm::vec3 intersect2 = submergedPoint2 + t2 * (abovePoint - submergedPoint2);
        
        // 计算水上小三角形面积
        glm::vec3 edge1 = intersect1 - abovePoint;
        glm::vec3 edge2 = intersect2 - abovePoint;
        Scalar aboveArea = 0.5f * glm::length(glm::cross(edge1, edge2));
        
        return originalArea - aboveArea;
    }
}

void HybridPropellerManager::setSetpoint(Scalar s)
{
    if (inv_) s *= Scalar(-1);
    setpoint_ = normalized_ ? btClamped(s, Scalar(-1), Scalar(1)) : btClamped(s, -thruster_->getSetpointLimit(), thruster_->getSetpointLimit());
    thruster_->setSetpoint(setpoint_);
    Scalar limit = thruster_->getSetpointLimit();
    Scalar normalsizedsetpoint;
    if(normalized_){
        normalsizedsetpoint = btClamped(s, Scalar(-1), Scalar(1));
    }
    else{
        normalsizedsetpoint=btClamped(s/limit,Scalar(-1),Scalar(1));
    }
    propeller_->setSetpoint(normalsizedsetpoint);
}

Scalar HybridPropellerManager::getSetpoint() const
{
    return inv_ ? -setpoint_ : setpoint_;
}

Scalar HybridPropellerManager::getAngle() const
{
    return currentInWater_ ? thruster_->getAngle() : propeller_->getAngle();
}

Scalar HybridPropellerManager::getOmega() const
{
   return currentInWater_ ? thruster_->getOmega() : propeller_->getOmega();
}

Scalar HybridPropellerManager::getThrust() const
{
    return thrust_;
}

Scalar HybridPropellerManager::getTorque() const
{
    return torque_;
}

bool HybridPropellerManager::isPropellerRight() const
{
    return RH_;
}

void HybridPropellerManager::Update(Scalar dt)
{
    Actuator::Update(dt);
    if (attach == nullptr)
        return;

    // 获取环境
    SimulationManager* sim = SimulationApp::getApp()->getSimulationManager();
    Ocean* ocn = sim->getOcean();
    Atmosphere* atm = sim->getAtmosphere();
    Transform propTrans = attach->getOTransform() * o2a;
    Scalar submergedRatio = Scalar(0);
    bool currentInWater = false;

    // ================== 基于三角面片的精确表面积计算 ==================
    if (ocn != nullptr && solid_ != nullptr && faceMeshPrecomputed_)
    {
        const Mesh* mesh = solid_->getPhysicsMesh();
        if (mesh != nullptr)
        {
            const size_t numFaces = mesh->faces.size();
            
            // 优化1：预计算变换矩阵，避免在循环中重复计算
            Transform meshTrans = propTrans;
            const glm::mat4 meshMatrix = glMatrixFromTransform(meshTrans);
            
            // ================== 第一阶段：快速检查是否完全浸没 ==================
            bool fullySubmerged = true;
            bool hasPartialSubmersion = false;
            
            // 使用并行检查完全浸没状态
            const unsigned int numThreads = std::max(1u, std::thread::hardware_concurrency());
            const size_t chunkSize = std::max(size_t(1), numFaces / numThreads);
            
            std::vector<std::future<std::pair<bool, bool>>> checkFutures;
            checkFutures.reserve(numThreads);
            
            for (unsigned int t = 0; t < numThreads; ++t)
            {
                size_t startIdx = t * chunkSize;
                size_t endIdx = (t == numThreads - 1) ? numFaces : std::min((t + 1) * chunkSize, numFaces);
                
                if (startIdx >= numFaces) break;
                
                checkFutures.emplace_back(std::async(std::launch::async, 
                    [this, mesh, &meshMatrix, ocn, startIdx, endIdx]() -> std::pair<bool, bool>
                    {
                        bool localFullySubmerged = true;
                        bool localHasPartialSubmersion = false;
                        
                        for (size_t i = startIdx; i < endIdx; ++i)
                        {
                            // 获取三角形的三个顶点（局部坐标）
                            const glm::vec3 p1gl = mesh->getVertexPos(i, 0);
                            const glm::vec3 p2gl = mesh->getVertexPos(i, 1);
                            const glm::vec3 p3gl = mesh->getVertexPos(i, 2);
                            
                            // 变换到世界坐标系
                            const glm::vec3 p1 = glm::vec3(meshMatrix * glm::vec4(p1gl, 1.f));
                            const glm::vec3 p2 = glm::vec3(meshMatrix * glm::vec4(p2gl, 1.f));
                            const glm::vec3 p3 = glm::vec3(meshMatrix * glm::vec4(p3gl, 1.f));
                            
                            // 检查深度
                            const GLfloat depth1 = ocn->GetDepth(p1);
                            const GLfloat depth2 = ocn->GetDepth(p2);
                            const GLfloat depth3 = ocn->GetDepth(p3);
                            
                            // 计算浸没顶点数量
                            int submergedCount = 0;
                            if (depth1 >= 0.f) submergedCount++;
                            if (depth2 >= 0.f) submergedCount++;
                            if (depth3 >= 0.f) submergedCount++;
                            
                            // 检查是否完全浸没
                            if (submergedCount < 3)
                            {
                                localFullySubmerged = false;
                            }
                            
                            // 检查是否有部分浸没
                            if (submergedCount > 0 && submergedCount < 3)
                            {
                                localHasPartialSubmersion = true;
                            }
                            
                            // 早期退出优化：如果已经确定不是完全浸没且有部分浸没，可以提前结束
                            if (!localFullySubmerged && localHasPartialSubmersion)
                                break;
                        }
                        
                        return std::make_pair(localFullySubmerged, localHasPartialSubmersion);
                    }));
            }
            
            // 收集第一阶段检查结果
            for (auto& future : checkFutures)
            {
                auto result = future.get();
                if (!result.first) fullySubmerged = false;
                if (result.second) hasPartialSubmersion = true;
            }
            
            // ================== 第二阶段：根据检查结果决定计算策略 ==================
            if (fullySubmerged)
            {
                // 完全浸没，直接设置比例为1
                submergedRatio = Scalar(1);
                currentInWater = true;
            }
            else if (!hasPartialSubmersion)
            {
                // 完全在水面上，直接设置比例为0
                submergedRatio = Scalar(0);
                currentInWater = false;
            }
            else
            {
                // 部分浸没，进行精确的面积累加计算
                std::vector<std::future<Scalar>> areaFutures;
                areaFutures.reserve(numThreads);
                
                for (unsigned int t = 0; t < numThreads; ++t)
                {
                    size_t startIdx = t * chunkSize;
                    size_t endIdx = (t == numThreads - 1) ? numFaces : std::min((t + 1) * chunkSize, numFaces);
                    
                    if (startIdx >= numFaces) break;
                    
                    areaFutures.emplace_back(std::async(std::launch::async, 
                        [this, mesh, &meshMatrix, ocn, startIdx, endIdx]() -> Scalar
                        {
                            Scalar localSubmergedArea = Scalar(0);
                            
                            for (size_t i = startIdx; i < endIdx; ++i)
                            {
                                // 获取三角形的三个顶点（局部坐标）
                                const glm::vec3 p1gl = mesh->getVertexPos(i, 0);
                                const glm::vec3 p2gl = mesh->getVertexPos(i, 1);
                                const glm::vec3 p3gl = mesh->getVertexPos(i, 2);
                                
                                // 变换到世界坐标系 - 使用预计算的矩阵
                                const glm::vec3 p1 = glm::vec3(meshMatrix * glm::vec4(p1gl, 1.f));
                                const glm::vec3 p2 = glm::vec3(meshMatrix * glm::vec4(p2gl, 1.f));
                                const glm::vec3 p3 = glm::vec3(meshMatrix * glm::vec4(p3gl, 1.f));
                                
                                // 批量检查深度，减少虚函数调用
                                const GLfloat depth1 = ocn->GetDepth(p1);
                                const GLfloat depth2 = ocn->GetDepth(p2);
                                const GLfloat depth3 = ocn->GetDepth(p3);
                                
                                // 早期退出优化：如果三角面片完全在水面上，跳过
                                if (depth1 < 0.f && depth2 < 0.f && depth3 < 0.f)
                                    continue;
                                
                                // 使用预计算的面积和优化的面积计算函数
                                const Scalar triangleSubmergedArea = calculateTriangleSubmergedArea(
                                    p1, p2, p3, depth1, depth2, depth3, triangleAreas_[i]);
                                
                                localSubmergedArea += triangleSubmergedArea;
                            }
                            
                            return localSubmergedArea;
                        }));
                }
                
                // 收集所有线程的结果
                Scalar submergedSurfaceArea = Scalar(0);
                for (auto& future : areaFutures)
                {
                    submergedSurfaceArea += future.get();
                }
                
                // 计算浸没表面积比例
                if (totalSurfaceArea_ > 1e-9f)
                {
                    submergedRatio = submergedSurfaceArea / totalSurfaceArea_;
                    submergedRatio = btClamped(submergedRatio, Scalar(0), Scalar(1));
                }
                
                // 根据浸没比例判断当前状态
                currentInWater = submergedRatio > Scalar(0.5);
            }
        }
        else
        {
            // 没有网格时，使用简单的点检测
            currentInWater = ocn->IsInsideFluid(propTrans.getOrigin());
            submergedRatio = currentInWater ? Scalar(1) : Scalar(0);
        }
    }
    else
    {
        // 没有海洋环境时，默认在空气中
        currentInWater = false;
        submergedRatio = Scalar(0);
    }

    // 更新当前状态
    currentInWater_ = currentInWater;

    // ================== 物理精确的推力和力矩计算 ==================
    thruster_->setSetpoint(setpoint_);
    propeller_->setSetpoint(setpoint_);
    thruster_->Update(dt);
    propeller_->Update(dt);

    const Scalar thrusterThrust = thruster_->getThrust();
    const Scalar thrusterTorque = thruster_->getTorque();
    const Scalar propellerThrust = propeller_->getThrust();
    const Scalar propellerTorque = propeller_->getTorque();
    
    // 密度感知推力混合
    const Scalar rho_water = ocn ? ocn->getLiquid().density : 1000.0;
    const Scalar rho_air = atm ? atm->getGas().density : 1.225;
    const Scalar densityRatio = rho_water / rho_air;
    const Scalar effectiveRatio = btMin(submergedRatio * densityRatio, 1.0);
    thrust_ = (effectiveRatio * thrusterThrust) + ((1.0 - effectiveRatio) * propellerThrust);
    
    // 基于功率平衡的扭矩混合
    const Scalar omega_water = thruster_->getOmega();
    const Scalar omega_air = propeller_->getOmega();
    const Scalar blendedOmega = submergedRatio * omega_water + (1.0 - submergedRatio) * omega_air;
    
    const Scalar torqueScale = (blendedOmega > 1e-6) ? 
        ((submergedRatio * thrusterTorque * omega_water) + 
         ((1.0 - submergedRatio) * propellerTorque * omega_air)) / blendedOmega : 0.0;
    
    torque_ = torqueScale * (RH_ ? Scalar(1) : Scalar(-1));

    // 施加推力和力矩
    if (thrust_ != Scalar(0) || torque_ != Scalar(0))
    {
        const Vector3 thrustV(thrust_, 0, 0);
        const Vector3 torqueV(torque_, 0, 0);
        const Transform solidTrans = attach->getCGTransform();
        attach->ApplyCentralForce(propTrans.getBasis() * thrustV);
        attach->ApplyTorque((propTrans.getOrigin() - solidTrans.getOrigin()).cross(propTrans.getBasis() * thrustV));
        attach->ApplyTorque(propTrans.getBasis() * torqueV);
    }
}

std::vector<Renderable> HybridPropellerManager::Render()
{
    Transform propTrans = Transform::getIdentity();
    if (attach != nullptr)
        propTrans = attach->getOTransform() * o2a;
    else
        return LinkActuator::Render();

    // 获取当前浸没比例
    Scalar immersionRatio = thrust_ ? (thruster_->getThrust() / thrust_) : 0.0;
    
    // 基于浸没比例旋转螺旋桨
    Scalar waterAngle = thruster_->getAngle();
    Scalar airAngle = propeller_->getAngle();
    Scalar blendedAngle = immersionRatio * waterAngle + (1.0 - immersionRatio) * airAngle;
    propTrans *= Transform(Quaternion(0, 0, blendedAngle), Vector3(0, 0, 0));

    // 添加可渲染对象
    std::vector<Renderable> items;
    items.reserve(2); // 预分配空间
    
    Renderable item;
    item.type = RenderableType::SOLID;
    item.materialName = solid_->getMaterial().name;
    item.objectId = solid_->getGraphicalObject();
    item.lookId = dm == DisplayMode::GRAPHICAL ? solid_->getLook() : -1;
    item.model = glMatrixFromTransform(propTrans);
    items.push_back(item);

    // 添加推力可视化
    Renderable thrustIndicator;
    thrustIndicator.type = RenderableType::ACTUATOR_LINES;
    thrustIndicator.model = item.model;
    thrustIndicator.points.reserve(2); // 预分配
    thrustIndicator.points.push_back(glm::vec3(0, 0, 0));
    thrustIndicator.points.push_back(glm::vec3(0.1f * thrust_, 0, 0));
    
    // 使用 cor 成员替代 color
    const Scalar blue = immersionRatio;
    const Scalar red = 1.0 - immersionRatio;
    thrustIndicator.cor = glm::vec4(red, 0, blue, 1);
    items.push_back(thrustIndicator);
    
    return items;
}

void HybridPropellerManager::WatchdogTimeout()
{
    setSetpoint(Scalar(0));
}

void HybridPropellerManager::AttachToSolid(SolidEntity* solid, const Transform& tf)
{
    LinkActuator::AttachToSolid(solid, tf);
    thruster_->AttachToSolid(solid, tf);
    propeller_->AttachToSolid(solid, tf);
}

bool HybridPropellerManager::getCurrentInWater() const
{
    return currentInWater_;
}

} // namespace sf