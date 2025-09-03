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

#ifndef __Stonefish_HybridPropellerManager__
#define __Stonefish_HybridPropellerManager__

#include "actuators/LinkActuator.h"
#include "actuators/Thruster.h"
#include "actuators/Propeller.h"
#include "entities/SolidEntity.h"
#include "entities/forcefields/Atmosphere.h"
#include "entities/forcefields/Ocean.h"

namespace sf
{
    class HybridPropellerManager : public LinkActuator
    {
    public:
        // 构造函数
        HybridPropellerManager(std::string uniqueName,
                              std::shared_ptr<SolidEntity> propeller,
                              Scalar diameter,
                              Scalar thrustCoeffAir, Scalar torqueCoeffAir, Scalar maxRPMAir,
                              std::shared_ptr<RotorDynamics> rotorDynamicsWater,
                              std::shared_ptr<ThrustModel> thrustModelWater,
                              Scalar maxSetpointWater,
                              bool rightHand, bool inverted = false, bool normalizedSetpoint = true
                              ); 

        // 控制和状态访问
        void setSetpoint(Scalar s);
        Scalar getSetpoint() const;
        Scalar getAngle() const;
        Scalar getOmega() const;
        Scalar getThrust() const;
        Scalar getTorque() const;
        bool isPropellerRight() const; 
        bool getCurrentInWater() const;

        // 核心方法
        void Update(Scalar dt) override;
        std::vector<Renderable> Render() override;
        void WatchdogTimeout() override;
        ActuatorType getType() const override {
            return inWater_ ? ActuatorType::THRUSTER : ActuatorType::PROPELLER;
        }
        void AttachToSolid(sf::SolidEntity* solid, const Transform& tf);

     private:
        void precomputeMeshAreas();
         /*!
        \param p1, p2, p3 三角形的三个顶点（世界坐标）
        \param depth1, depth2, depth3 三个顶点的水深
        \param originalArea 原始三角形面积
        \return 浸没部分的面积
         */
        Scalar calculateTriangleSubmergedArea(
            const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3,
            GLfloat depth1, GLfloat depth2, GLfloat depth3,
            Scalar originalArea) const;
        std::shared_ptr<Thruster> thruster_; // 水下推进器
        std::shared_ptr<Propeller> propeller_; // 空中螺旋桨
        std::shared_ptr<SolidEntity> solid_; // 新增：存储 propeller 参数
        bool inWater_; // 当前是否在水中
        bool RH_; // 是否右旋螺旋桨
        bool inv_; // 是否反转设定点
        bool normalized_; // 是否归一化设定点
        Scalar setpoint_; // 当前设定点
        Scalar thrust_; // 当前推力
        Scalar torque_; // 当前力矩
        Scalar transitionFactor_;   // 过渡因子（0~1）
        bool targetInWater_;        // 目标环境状态
        bool currentInWater_;
        bool faceMeshPrecomputed_;              // 是否已预计算网格
        std::vector<Scalar> triangleAreas_;     // 预计算的三角面片面积
        Scalar totalSurfaceArea_;               // 总表面积

    };
}

#endif