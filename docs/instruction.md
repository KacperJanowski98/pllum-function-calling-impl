# Fine-tutnig PLLuM

The project should be expanded to include the functionality of performing fine-tuning of the Polish large language model PLLuM 8B instruct using the dataset created in the project with examples of function calling in both Polish and English:

**[CYFRAGOVPL/Llama-PLLuM-8B-instruct](https://huggingface.co/CYFRAGOVPL/Llama-PLLuM-8B-instruct)** - Polish government-backed Llama-based model

This is an 8B model, but due to the hardware limitations of the Nvida RTX 4060 GPU, QLoRA should be used for fine tuning with appropriate settings for this equipment.

## Requirements

- I would like to use pytorch with GPU (cuda) for fine tuning. Use QLoRA 4bit and a model corrected with hugging face. If possible, use the Unsloth framework for fine tuning. Link to the Unsloth repository [Unsloth](https://github.com/unslothai/unsloth)
- Add the required dependencies to pyproject.toml
- Prepare the appropriate methods according to the project structure
- Provide the ability to configure fine-tuning parameters
- Use a data set, with translated examples (data saved in the data folder)
- Fine tuning process carried out in a dedicated notebook
- The resulting model saved in the models folder
- The second notebook is for testing the model's operation with function calling after the fine-tuning process (loading the model from the models folder)
- Update README.md

## Repeating the dataset structure

The JSON data consists of the following key-value pairs:

- `query` (string): The query or problem statement.
- `tools` (array): An array of available tools that can be used to solve the query.
    - Each tool is represented as an object with the following properties:
        - `name` (string): The name of the tool.
        - `description` (string): A brief description of what the tool does.
        - `parameters` (object): An object representing the parameters required by the tool.
            - Each parameter is represented as a key-value pair, where the key is the parameter name and the value is an object with the following properties:
                - `type` (string): The data type of the parameter (e.g., "int", "float", "list").
                - `description` (string): A brief description of the parameter.
                - `required` (boolean): Indicates whether the parameter is required or optional.
- `answers` (array): An array of answers corresponding to the query.
    - Each answer is represented as an object with the following properties:
        - `name` (string): The name of the tool used to generate the answer.
        - `arguments` (object): An object representing the arguments passed to the tool to generate the answer.
            - Each argument is represented as a key-value pair, where the key is the parameter name and the value is the corresponding value.
