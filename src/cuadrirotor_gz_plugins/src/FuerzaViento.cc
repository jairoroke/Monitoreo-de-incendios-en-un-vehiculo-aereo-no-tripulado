// Sistema de Gazebo que aplica sobre un link una fuerza externa en el marco
// inercial. La fuerza se mantiene en cada paso de física hasta que llega un
// nuevo valor por el tópico, de modo que la perturbación es exactamente la
// señal publicada (retenida entre mensajes), sin acumularse.
//
// Uso en el SDF del modelo:
//   <plugin filename="FuerzaViento" name="cuadrirotor::FuerzaViento">
//     <link_name>base_link</link_name>
//     <topic>/cuadrirotor/viento</topic>
//   </plugin>
#include <mutex>
#include <string>

#include <gz/math/Vector3.hh>
#include <gz/msgs/vector3d.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/transport/Node.hh>

namespace cuadrirotor
{
class FuerzaViento : public gz::sim::System,
                     public gz::sim::ISystemConfigure,
                     public gz::sim::ISystemPreUpdate
{
public:
  void Configure(const gz::sim::Entity &_entity,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 gz::sim::EntityComponentManager &_ecm,
                 gz::sim::EventManager &) override
  {
    gz::sim::Model modelo(_entity);
    const auto nombre = _sdf->Get<std::string>("link_name", "base_link").first;
    this->link = gz::sim::Link(modelo.LinkByName(_ecm, nombre));
    if (!this->link.Valid(_ecm))
    {
      gzerr << "FuerzaViento: no existe el link [" << nombre << "]\n";
      return;
    }
    const auto topico = _sdf->Get<std::string>(
      "topic", "/model/" + modelo.Name(_ecm) + "/viento").first;
    this->nodo.Subscribe(topico, &FuerzaViento::AlRecibir, this);
    gzmsg << "FuerzaViento: link [" << nombre << "], tópico [" << topico << "]\n";
  }

  void PreUpdate(const gz::sim::UpdateInfo &_info,
                 gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused || !this->link.Valid(_ecm))
      return;
    gz::math::Vector3d f;
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      f = this->fuerza;
    }
    if (f != gz::math::Vector3d::Zero)
      this->link.AddWorldForce(_ecm, f);
  }

private:
  void AlRecibir(const gz::msgs::Vector3d &_msg)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->fuerza.Set(_msg.x(), _msg.y(), _msg.z());
  }

  gz::sim::Link link{gz::sim::kNullEntity};
  gz::transport::Node nodo;
  std::mutex mutex;
  gz::math::Vector3d fuerza{gz::math::Vector3d::Zero};
};
}  // namespace cuadrirotor

GZ_ADD_PLUGIN(cuadrirotor::FuerzaViento, gz::sim::System,
              cuadrirotor::FuerzaViento::ISystemConfigure,
              cuadrirotor::FuerzaViento::ISystemPreUpdate)
