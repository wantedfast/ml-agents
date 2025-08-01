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
* [Python APIs]()
  * [Python Gym API](Python-Gym-API.md)
  * [Python PettingZoo API](Python-PettingZoo-API.md)
  * [Python Low-Level API](Python-LLAPI.md)
* [Advanced Features]()
  * [Custom Side Channels](Custom-SideChannels.md)
  * [Inference Engine](Inference-Engine.md)
  * [Hugging Face Integration](Hugging-Face-Integration.md)
  * [Custom Grid Sensors](Custom-GridSensors.md)
  * [Input System Integration](InputSystem-Integration.md)
* [Cloud & Deployment (deprecated)]()
  * [Using Docker](Using-Docker.md)
  * [Amazon Web Services](Training-on-Amazon-Web-Service.md)
  * [Microsoft Azure](Training-on-Microsoft-Azure.md)
* [Reference & Support]()
  * [FAQ](FAQ.md)
  * [Limitations](Limitations.md)
  * [Migrating](Migrating.md)
  * [Background: Machine Learning](Background-Machine-Learning.md)
  * [Background: Unity](Background-Unity.md)
  * [Background: PyTorch](Background-PyTorch.md)

## Next Steps

| [Making a New Learning Environment](Learning-Environment-Create-New.md) | Create your own Learning Environment.   |

- For more information on the ML-Agents Toolkit, in addition to helpful
  background, check out the [ML-Agents Toolkit Overview](ML-Agents-Overview.md)
  page.

- For more information on the various training options available, check out the
  [Training ML-Agents](Training-ML-Agents.md) page.

[the Agent documentation](Learning-Environment-Design-Agents.md#decisions)
Hyperparameters are explained in [the training configuration file documentation](Training-Configuration-File.md)

## Help

If you run into any problems regarding ML-Agents, refer to our [FAQ](FAQ.md) and
our [Limitations](Limitations.md) pages. If you can't find anything please
[submit an issue](https://github.com/Unity-Technologies/ml-agents/issues) and
make sure to cite relevant information on OS, Python version, and exact error
message (whenever possible).
package
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

### For a broad overview of reinforcement learning, imitation learning and all the
training scenarios, methods and options within the ML-Agents Toolkit, see
[ML-Agents Toolkit Overview](ML-Agents-Overview.md).

[Inference Engine](Inference-Engine.md)


## Summary and Next Steps

To briefly summarize: The ML-Agents Toolkit enables games and simulations built
in Unity to serve as the platform for training intelligent agents. It is
designed to enable a large variety of training modes and scenarios and comes
packed with several features to enable researchers and developers to leverage
(and enhance) machine learning within Unity.

In terms of next steps:

- For a walkthrough of running ML-Agents with a simple scene, check out the
  [Getting Started](Sample.md) guide.
- For a "Hello World" introduction to creating your own Learning Environment,
  check out the
  [Making a New Learning Environment](Learning-Environment-Create-New.md) page.
- For an overview on the more complex example environments that are provided in
  this toolkit, check out the
  [Example Environments](Learning-Environment-Examples.md) page.
- For more information on the various training options available, check out the
  [Training ML-Agents](Training-ML-Agents.md) page.
