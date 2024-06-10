# iModulon Chatbot Assistant
This Jupyter Notebook contains an AI chatbot assistant designed to interact with the [iModulon database](https://imodulondb.org/about.html). If you are a biologist interested in what machine learning can tell us about the regulation of bacterial gene expression, this site will provide very valuable tools for you. The chatbot utilizes OpenAI's GPT-4o model and langchain to answer queries, provide information, and assist with data analysis related to the iModulons of Escherichia coli. The assistant also has access to gene information from the [EcoCyc](https://ecocyc.org/) database.


## Functionality
The chatbot supports a variety of functions and tools designed to facilitate using iModulons, including but not limited to:

* Learning about iModulons

* Finding iModulon, genes, and conditions

* Getting detailed information about genes from EcoCyc and conditions

* Plotting gene expression and iModulon activity

* Comparing gene expression and iModulon activities


## Notebook Overview
The [notebook](imodulon_chat_assistant.ipynb) is structured as follows:

Imports and Setup: Includes necessary imports and configurations for running the chatbot.

Environment Setup: Input your OpenAI API key to access the GPT-4o model.

Chatbot Initialization: Sets up the chatbot, loads necessary tools, and defines the chat prompt template.

Chat Interface: A simple interface for interacting with the chatbot, where users can input queries and receive responses. 

For detailed examples demonstrating the capabilities of the chatbot, refer to the [example conversations](example_chats/).


## Installation
Environment Setup
The required dependencies are listed in environment.yml and requirements.txt. You can set up the environment using Conda or pip.

Using [Conda](https://anaconda.org/):

` conda env create -f environment.yml `

`conda activate imodulon`

Using [pip](https://pypi.org/project/pymodulon/):

`pip install -r requirements.txt`

## Contact
For any questions or issues, please contact Luis at lsanchezdelavega@ucsd.edu
