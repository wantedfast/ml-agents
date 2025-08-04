* [ML-Agents package](index.md)
* [ML-Agents Theory](ML-Agents-Overview.md)
* [Get started](Get-Started.md)
  * [Installation](Installation.md)
  * [Sample: Running an Example Environment](Sample.md)
  * [More Example Environments](Learning-Environment-Examples.md)
* [Learning Environments and Agents](Learning-Environments-Agents.md)
    * [Designing Learning Environments](Learning-Environment-Design.md)
    * [Designing Agents](Learning-Environment-Design-Agents.md)
    * [Sample: Making a New Learning Environment](Learning-Environment-Create-New.md)
* [Training](Training.md)
  * [Training ML-Agents Basics](Training-ML-Agents.md)
  * [Training Configuration File](Training-Configuration-File.md)
  * [Using Tensorboard](Using-Tensorboard.md)
* [Python APIs](Python-APIs.md)
  * [Python Gym API](Python-Gym-API.md)
  * [Python PettingZoo API](Python-PettingZoo-API.md)
  * [Python Low-Level API](Python-LLAPI.md)
* [Python Tutorial with Google Colab](Tutorial-Colab.md)
  * [Using a UnityEnvironment](https://colab.research.google.com/github/Unity-Technologies/ml-agents/blob/release_22_docs/colab/Colab_UnityEnvironment_1_Run.ipynb)
  * [Q-Learning with a UnityEnvironment](https://colab.research.google.com/github/Unity-Technologies/ml-agents/blob/release_22_docs/colab/Colab_UnityEnvironment_2_Train.ipynb)
  * [Using Side Channels on a UnityEnvironment](https://colab.research.google.com/github/Unity-Technologies/ml-agents/blob/release_22_docs/colab/Colab_UnityEnvironment_3_SideChannel.ipynb)
* [Advanced Features](Advanced-Features.md)
  * [Custom Side Channels](Custom-SideChannels.md)
  * [Custom Grid Sensors](Custom-GridSensors.md)
  * [Input System Integration](InputSystem-Integration.md)
  * [Inference Engine](Inference-Engine.md)
  * [Hugging Face Integration](Hugging-Face-Integration.md)
* [Cloud & Deployment (deprecated)](Cloud-Deployment.md)
  * [Using Docker](Using-Docker.md)
  * [Amazon Web Services](Training-on-Amazon-Web-Service.md)
  * [Microsoft Azure](Training-on-Microsoft-Azure.md)
* [Reference & Support](Reference-Support.md)
  * [FAQ](FAQ.md)
  * [Limitations](Limitations.md)
  * [Migrating](Migrating.md)
  * [versioning](Versioning.md)
  * [ML-Agents Glossary](Glossary.md)
* [Background](Background.md)
  * [Machine Learning](Background-Machine-Learning.md)
  * [Unity](Background-Unity.md)
  * [PyTorch](Background-PyTorch.md)
  * [ELO](ELO-Rating-System.md)

## Next Steps


multienv vs multi instances
Using an Executable Environment
ML-Agents Package Settings
Profiling Trainers
How to use the Unity Environment Registry
Python API Documentation
Python-On-Off-Policy-Trainer-Documentation
Custom Trainer Plugin
Customizing Training via Plugins


[the Agent documentation](Learning-Environment-Design-Agents.md#decisions)
Hyperparameters are explained in [the training configuration file documentation](Training-Configuration-File.md)


## Capabilities
The package allows you to convert any Unity scene into a learning environment and train character behaviors using a variety of machine-learning algorithms. Additionally, it allows you to embed these trained behaviors back into Unity scenes to control your characters. More specifically, the package provides the following core functionalities:

* Define Agents: entities, or characters, whose behavior will be learned. Agents are entities that generate observations (through sensors), take actions, and receive rewards from the environment.
* Define Behaviors: entities that specify how an agent should act. Multiple agents can share the same Behavior and a scene may have multiple Behaviors.
* Record demonstrations: To show the behaviors of an agent within the Editor. You can use demonstrations to help train a behavior for that agent.
* Embed a trained behavior (aka: run your ML model) in the scene via the [Inference Engine](https://docs.unity3d.com/Packages/com.unity.ai.inference@latest). Embedded behaviors allow you to switch an Agent between learning and inference.

###
Note: You can train using an executable rather than the Editor. To do so, follow the instructions in Using an Executable.

To help on-board to the entire set of functionality provided by the ML-Agents
Toolkit, we recommend exploring our [API documentation](API-Reference.md).
Additionally, our [example environments](Learning-Environment-Examples.md) are a
great resource as they provide sample usage of almost all of our features.

### * [Training Plugins](Training-Plugins.md)






