// Sistema de Gazebo que junta comandos de motor sueltos (un gz.msgs.Double por
// motor, como los publica ArduPilotPlugin con <type>COMMAND</type>) en un solo
// gz.msgs.Actuators, que es lo que espera MulticopterMotorModel. Así un
// controlador externo (ArduPilot SITL) mueve los mismos motores calibrados que
// usan los controladores de ROS 2.
//
// Uso en el SDF del modelo:
//   <plugin filename="ComandosMotores" name="cuadrirotor::ComandosMotores">
//     <entrada>/cuadrirotor/ap/motor_0</entrada>   (una por motor, en orden)
//     ...
//     <salida>/cuadrirotor/command/motor_speed</salida>
//   </plugin>
#include <mutex>
#include <string>
#include <vector>

#include <gz/msgs/actuators.pb.h>
#include <gz/msgs/double.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>

namespace cuadrirotor
{
class ComandosMotores : public gz::sim::System,
                        public gz::sim::ISystemConfigure,
                        public gz::sim::ISystemPreUpdate
{
public:
  void Configure(const gz::sim::Entity &,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 gz::sim::EntityComponentManager &,
                 gz::sim::EventManager &) override
  {
    auto elem = std::const_pointer_cast<sdf::Element>(_sdf);
    for (auto e = elem->FindElement("entrada"); e; e = e->GetNextElement("entrada"))
    {
      const auto topico = e->Get<std::string>();
      const size_t i = this->velocidades.size();
      this->velocidades.push_back(0.0);
      std::function<void(const gz::msgs::Double &)> cb =
        [this, i](const gz::msgs::Double &_msg)
        {
          std::lock_guard<std::mutex> lock(this->mutex);
          this->velocidades[i] = _msg.data();
          this->nuevo = true;
        };
      this->nodo.Subscribe(topico, cb);
    }
    const auto salida = _sdf->Get<std::string>("salida", "command/motor_speed").first;
    this->pub = this->nodo.Advertise<gz::msgs::Actuators>(salida);
    gzmsg << "ComandosMotores: " << this->velocidades.size() << " entradas -> ["
          << salida << "]\n";
  }

  void PreUpdate(const gz::sim::UpdateInfo &_info,
                 gz::sim::EntityComponentManager &) override
  {
    if (_info.paused)
      return;
    gz::msgs::Actuators msg;
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      if (!this->nuevo)
        return;
      this->nuevo = false;
      for (double w : this->velocidades)
        msg.add_velocity(w);
    }
    this->pub.Publish(msg);
  }

private:
  gz::transport::Node nodo;
  gz::transport::Node::Publisher pub;
  std::mutex mutex;
  std::vector<double> velocidades;
  bool nuevo{false};
};
}  // namespace cuadrirotor

GZ_ADD_PLUGIN(cuadrirotor::ComandosMotores, gz::sim::System,
              cuadrirotor::ComandosMotores::ISystemConfigure,
              cuadrirotor::ComandosMotores::ISystemPreUpdate)
