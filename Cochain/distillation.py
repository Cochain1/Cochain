import sys
import json
from api_provider import AIProvider
from tqdm import tqdm

def load_prompt_template(template_path):
    try:
        with open(template_path, "r", encoding="UTF-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Template file {template_path} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading template: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Load prompt template
    PROMPT_TEMPLATE = load_prompt_template("prompt_template.txt")
    
    # Read JSON file
    with open("data.json", "r", encoding="UTF-8") as file:
        content = json.load(file)

    with open("error_instruction.txt", "a", encoding="UTF-8") as file2, \
         open("design_produce_method.json", "a", encoding="UTF-8") as file4:
        
        with tqdm(total=len(content), desc="Processing", unit="item") as pbar:
            for data in content:
                try:
                    input_text = data.get("input", "")
                    output_text = data.get("output", "")
                    original_index = data.get("index")

                    # Format the prompt template
                    formatted_prompt = PROMPT_TEMPLATE.format(
                        input_text=input_text,
                        output_text=output_text
                    )

                    client = AIProvider(api_key="YOUR_API_KEY")
                    messages = [{
                        "role": "user",
                        "content": formatted_prompt
                    }]

                    response = client.chat.completions.create(
                        model="",
                        messages=messages,
                    )
                    
                    methods_text = response.choices[0].message.content
                    produce_method = methods_text.replace("methods:", "").strip()

                    design_produce_data = {
                        "index": original_index,
                        "design_method": input_text,
                        "produce_method": produce_method
                    }
                    file4.write(json.dumps(design_produce_data, ensure_ascii=False) + ",\n")
                    file4.flush()
                
                except Exception as e:
                    print(f"{original_index} failed: {input_text}")
                    print(e)
                    file2.write(f"{original_index}: {input_text}\n")
                    file2.flush()
                
                finally:
                    pbar.update(1)