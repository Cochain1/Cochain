import sys
import json
from api_provider import AIProvider

if __name__ == '__main__':
    # Read JSON file
    with open("car_design.json", "r", encoding="UTF-8") as file:
        content = json.load(file) 
    
    file2 = open("error_counterfactual.txt", "a", encoding="UTF-8")
    file3 = open("design_counterfactual.txt", "a", encoding="UTF-8")
    index = 0
    while index < 2500:
        try:
            data = content[index]
            instruction = data.get("instruction", "")

            # Generate different question transformations based on the type of counterfactual reasoning
            if index % 5 == 0:
                # Causal counterfactual reasoning
                example = "Causal counterfactual reasoning, assuming certain factors affecting the original problem have changed. For example, the original question is modified to: If engine design efficiency does not improve, will fuel economy increase?"
                instruction_modified = f"{example}"
            
            elif index % 5 == 1:
                # Opposite counterfactual reasoning
                example = "Opposite counterfactual reasoning, assuming the opposite action or decision. For example, the original question is modified to: If no fuel economy optimization measures are taken, how would fuel economy change?"
                instruction_modified = f"{example}"
            
            elif index % 5 == 2:
                # Alternative counterfactual reasoning
                example = "Alternative counterfactual reasoning, assuming the use of different methods or technologies, or a completely different solution. For example, the original question is modified to: If an electric powertrain is used instead of an internal combustion engine, would fuel economy still need to be improved?"
                instruction_modified = f"{example}"
            
            elif index % 5 == 3:
                # Extreme counterfactual reasoning
                example = "Extreme counterfactual reasoning, assuming all relevant conditions are pushed to their limits. For example, the original question is modified to: If engine efficiency reaches 100%, what level would fuel economy achieve?"
                instruction_modified = f"{example}"
            
            else:
                # Reverse causal reasoning
                example = "Reverse causal reasoning, assuming the original problem has already achieved the target value, deduce what conditions must be met. For example, the original question is modified to: If fuel economy significantly improves, what changes might have occurred in engine technology, body design, or driving habits?"
                instruction_modified = f"{example}"

            client = AIProvider(api_key="")
            messages = [{"role": "user",
                         "content": f"You are an expert in counterfactual reasoning. Please read an example first. The original question is: How to improve fuel economy? Now using {instruction_modified}\n"
                                    f"I now have a question and need to generate its counterfactual version. My question is: {instruction}\n"
                                    f"Please directly output its counterfactual version."}]
            response = client.chat.completions.create(
                model="",
                messages=messages,
            )
            
            response_text = response.choices[0].message.content
            print(f"{index+1} Success, API returned content: {response_text}")  # Add print to check the return value

            file3.write(f"{response_text}\n")

        except Exception as e:
            print(f"{index+1} Failed: {instruction}")
            print(e)
            file2.write(f"{instruction}\n")
        finally:
            index += 1

    file2.close()
    file3.close()