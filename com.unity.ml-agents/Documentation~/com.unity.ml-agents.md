# ML-Agents Overview
ML-agents enable games and simulations to serve as environments for training intelligent agents in Unity. Training can be done with reinforcement learning, imitation learning, neuroevolution, or any other methods. Trained agents can be used for many use cases, including controlling NPC behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release.

The _ML-Agents_ package has a C# SDK for the [Unity ML-Agents Toolkit], which can be used outside of Unity. This package now contains comprehensive documentation including detailed training guides, Python API documentation, and advanced features.

## Capabilities
The package allows you to convert any Unity scene into a learning environment and train character behaviors using a variety of machine-learning algorithms. Additionally, it allows you to embed these trained behaviors back into Unity scenes to control your characters. More specifically, the package provides the following core functionalities:

* Define Agents: entities, or characters, whose behavior will be learned. Agents are entities that generate observations (through sensors), take actions, and receive rewards from the environment.
* Define Behaviors: entities that specify how an agent should act. Multiple agents can share the same Behavior and a scene may have multiple Behaviors.
* Record demonstrations: To show the behaviors of an agent within the Editor. You can use demonstrations to help train a behavior for that agent.
* Embed a trained behavior (aka: run your ML model) in the scene via the [Unity Inference Engine]. Embedded behaviors allow you to switch an Agent between learning and inference.

## Special Notes
Note that the ML-Agents package does not contain the machine learning algorithms for training behaviors. The ML-Agents package only supports instrumenting a Unity scene, setting it up for training, and then embedding the trained model back into your Unity scene. The machine learning algorithms that orchestrate training are part of the companion Python package. For detailed training instructions, see [Training ML-Agents](Training-ML-Agents.md).

## Package contents

The following table describes the package folder structure:

| **Location**           | **Description**                                                         |
| ---------------------- | ----------------------------------------------------------------------- |
| _Documentation~_       | Contains the documentation for the Unity package.                       |
| _Editor_               | Contains utilities for Editor windows and drawers.                      |
| _Plugins_              | Contains third-party DLLs.                                              |
| _Runtime_              | Contains core C# APIs for integrating ML-Agents into your Unity scene.  |
| _Runtime/Integrations_ | Contains utilities for integrating ML-Agents into specific game genres. |
| _Tests_                | Contains the unit tests for the package.                                |

<a name="Installation"></a>

## Installation
To add the ML-Agents package to a Unity project:

* Create a new Unity project with Unity 6000.0 (or later) or open an existing one.
* To open the Package Manager, navigate to Window > Package Manager.
* Click + and select Add package by name...
* Enter com.unity.ml-agents
*Click Add to add the package to your project.

To install the companion Python package to enable training behaviors, follow the [Installation](Installation.md) instructions.

## Known Limitations

### Training

Training is limited to the Unity Editor and Standalone builds on Windows, MacOS,
and Linux with the Mono scripting backend. Currently, training does not work
with the IL2CPP scripting backend. Your environment will default to inference
mode if training is not supported or is not currently running.

### Inference

Inference is executed via [Unity Inference Engine](https://docs.unity3d.com/Packages/com.unity.ai.inference@latest) on the end-user device. Therefore, it is subject to the performance limitations of the end-user CPU or GPU. Also, only models created with our trainers are supported for running ML-Agents with a neural network behavior.

### Headless Mode

If you enable Headless mode, you will not be able to collect visual observations from your agents.

### Rendering Speed and Synchronization

Currently the speed of the game physics can only be increased to 100x real-time. The Academy (the sentinel that controls the stepping of the game to make sure everything is synchronized, from collection of observations to applying actions generated from policy inference to the agent) also moves in time with `FixedUpdate()` rather than `Update()`, so game behavior implemented in Update() may be out of sync with the agent decision-making.  See [Execution Order of Event Functions] for more information.

You can control the frequency of Academy stepping by calling `Academy.Instance.DisableAutomaticStepping()`, and then calling `Academy.Instance.EnvironmentStep()`.

## Complete Documentation

Welcome to the comprehensive Unity ML-Agents Toolkit documentation. This documentation is now fully integrated within the Unity package for easy access.

## Quick Navigation

### Getting Started
- **[Getting Started](Getting-Started.md)** - Step-by-step setup guide
- **[Installation](Installation.md)** - Complete installation instructions
- **[ML-Agents Overview](ML-Agents-Overview.md)** - Comprehensive feature overview

### Creating Learning Environments
- **[Learning Environment Design](Learning-Environment-Design.md)** - Design principles
- **[Creating New Learning Environments](Learning-Environment-Create-New.md)** - Step-by-step creation
- **[Designing Agents](Learning-Environment-Design-Agents.md)** - Agent design guide
- **[Learning Environment Examples](Learning-Environment-Examples.md)** - Example environments

### Training & Configuration
- **[Training ML-Agents](Training-ML-Agents.md)** - Complete training guide
- **[Training Configuration File](Training-Configuration-File.md)** - Configuration reference
- **[Training Plugins](Training-Plugins.md)** - Extending training functionality
- **[Using Tensorboard](Using-Tensorboard.md)** - Monitoring training

### Python APIs
- **[Python Gym API](Python-Gym-API.md)** - OpenAI Gym interface
- **[Python PettingZoo API](Python-PettingZoo-API.md)** - Multi-agent environments  
- **[Python Low-Level API](Python-LLAPI.md)** - Low-level API access

### Advanced Features
- **[Custom Side Channels](Custom-SideChannels.md)** - Custom communication
- **[Inference Engine](Inference-Engine.md)** - Running trained models
- **[Hugging Face Integration](Hugging-Face-Integration.md)** - Model sharing
- **[Custom Grid Sensors](Custom-GridSensors.md)** - Custom grid sensors
- **[Input System Integration](InputSystem-Integration.md)** - Unity Input System

### Cloud & Deployment
- **[Using Docker](Using-Docker.md)** - Containerized training
- **[Amazon Web Services](Training-on-Amazon-Web-Service.md)** - AWS deployment
- **[Microsoft Azure](Training-on-Microsoft-Azure.md)** - Azure deployment

### Reference & Support
- **[FAQ](FAQ.md)** - Frequently asked questions
- **[Limitations](Limitations.md)** - Known limitations
- **[Migrating](Migrating.md)** - Migration between versions
- **[Background: Machine Learning](Background-Machine-Learning.md)** - ML fundamentals
- **[Background: Unity](Background-Unity.md)** - Unity concepts
- **[Background: PyTorch](Background-PyTorch.md)** - PyTorch fundamentals

## Additional Resources

For the most up-to-date information and community support, visit our [GitHub repository](https://github.com/Unity-Technologies/ml-agents).

* [Unity Discussions](https://discussions.unity.com/tag/ml-agents)
* [Discord](https://discord.com/channels/489222168727519232/1202574086115557446)
* [Website](https://unity-technologies.github.io/ml-agents/)

[Unity Inference Engine]: https://docs.unity3d.com/Packages/com.unity.ai.inference@2.2/manual/index.html
[Execution Order of Event Functions]: https://docs.unity3d.com/Manual/ExecutionOrder.html

