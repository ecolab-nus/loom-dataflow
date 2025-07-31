# Scale Out
This sub projects aim at describing how componenets scales out.

# Basic Concepts
### Hardware Resources
The hardware is composed of **resources**. Each resource instance is an uniq and undividable.
A group of resources is a **Module**. For example, a memory module *can be modeled by* a composition of *ports* and *capacity*. This is of course one way of abstracting a memory module.

### Functionality and Materializations during compilation
A module, or a combination of modules can provide certain **Functionalities** for mapping **instructions**. Say in a different way, one or several instructions can be **materialized** with the modules. 

A **Functionality** describes: (1) the matching patterns to the compiler side, and (2) the hardware **modules** related to, and how the resources are removed from the modules if the matching instructions are **materialized** with this functionality.