import sys

class PromptLoader():
    def __init__(self, prompt_idx, video_type, label_type):
        self.prompt_idx = prompt_idx
        self.video_type = video_type
        self.label_type = label_type

    def apply(self, char_text):
        if self.prompt_idx == 0:
            general_prompt = (
                "Please watch the following {video_type} clip.\n"
                "Please briefly describe the what happened in the following three steps below:\n"
                "1. Identify main characters (if {label_type} are available){char_text};\n"
                "2. Describe the actions of characters, i.e., who is doing what, focusing on the movements;\n" 
                "3. Describe the interactions between characters, such as looking;\n"
                "Note, colored {label_type} are provided for character indications only, DO NOT mention them in the description.\n"   
                "Make sure you do not hallucinate information.\n"
                "### Answer Template ###\nDescription:\n1. Main characters: '';\n2. Actions: '';\n3. Character-character interactions: ''.\n\nExplanation: ''."
            ) 
        general_prompt = general_prompt.format(video_type=self.video_type, char_text=char_text, label_type=self.label_type)
        return general_prompt