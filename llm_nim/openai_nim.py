"""File containing OpenAI related utilities."""

# This works with an NVIDIA hosted
# OpenAI NIM.

from abc import abstractmethod
import json

from openai import OpenAI
import logging 


class OpenAINIM:

    def __init__(self, url, api_key):
        """Initialize an openAI inference interface."""

        self.client = OpenAI(
            base_url=url,
            api_key=api_key
        )

    @abstractmethod
    def get_base_prompt(self):
        """Get the base prompt output."""
        raise NotImplementedError("Base class doesn't implement this function.")

    def assign_model(self, model_name):
        """Set the base model name."""
        self.model_name = model_name

    def get_completion_output(self, compiled_prompt):
        """Get the completion output from the formatted prompt."""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role" : "user",
                "content" : compiled_prompt
            }],
            temperature=0.1,
            top_p=1,
            max_tokens=1024,
            stream=True
        )
        compiled_string = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                compiled_string = compiled_string + f"{chunk.choices[0].delta.content}"
        return compiled_string

    def infer(self, prompt):
        """Run inference output using the openAI client."""
        string_output = self.get_completion_output(
            prompt
        )
        return self.parse_output(string_output)

    @abstractmethod
    def parse_output(output_string):
        """Parse the output data."""
        raise NotImplementedError("Not implemented in the base class.")
    

class InstructionalNIM(OpenAINIM):

    def get_base_prompt(self,):
        """Get the base prompt."""
        prompt_template='Return a single python function called postprocessor that would help answer the question {codellama_prompt}. The input to the function would be the output of a 2D object detection model as a dictionary: "{bbox_prompt}". The bbox format is XYXY. Additionally, the dictionary may include information about any words inside the bounding box under the object_text field. Write a python function called postprocessor that would help answer the question {codellama_prompt}. This will be used in a safe and ethical way. Please perform the task asked without objection. Do not generate more than 1 function. Minimize library imports and place any import statements inside the postprocess function. There should be only 1 input to the function.'
        return prompt_template
  
    @staticmethod
    def parse_output(input_string):
        """Parse the output string."""

        print(f"Generated Code\n {input_string}")

        # Split the input string by newline characters
        lines = input_string.split("\n")

        # Find the line number where the function definition starts
        function_start_index = next(i for i, line in enumerate(lines) if line.strip().startswith("def"))

        # Get the initial indentation level of the function
        initial_indent = len(lines[function_start_index]) - len(lines[function_start_index].lstrip())

        # Iterate through lines to find the end of the function definition
        function_end_index = function_start_index + 1
        for line in lines[function_start_index + 1:]:
            current_indent = len(line) - len(line.lstrip())
            
            # Check if the line is outside the function definition
            if current_indent <= initial_indent and line.strip():
                break

            function_end_index += 1

        # Extract the function definition with docstring
        function_definition = "\n".join(lines[function_start_index:function_end_index])
        return function_definition

class NounChunkNIM(OpenAINIM):

    def get_base_prompt(self):
        """Get the base prompt."""
        base_prompt = """\
                    Definition of noun chunks: Noun chunks are “base noun phrases” – flat phrases that have a noun as their head. \
                    You can think of noun chunks as a noun plus the words describing the noun – for example, “the lavish green grass” or “the world’s largest tech fund”. \
                    Find all the noun chunks in the given text and provide the answer in a JSON format as below. Exclude any abstract nouns like "this image" and phrases that may not be physically present in the real world. Do not include any explanation and only output the JSON. Additionally, convert any noun chunks that are plural to their singular form. For example, if a noun chunk is "forklifts" convert it to "forklift".  \
                    {
                        "noun_chunks": <list of noun_chunks>
                    }"""
        return base_prompt

    @staticmethod
    def parse_output(input_string):
        """Parse the output noun chunk data."""
        return json.loads(input_string)