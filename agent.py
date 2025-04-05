# Create an agent that can evaluate some images

# first, start with simplest framework of averaging images....
# Note: assume access to file paths of images
import os 
import json 
import base64 
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class AverageEnsemble:
    def __init__(self, n_agents):
        self.n_agents = n_agents 

    def query_gpt_vision_for_ranking(self, image_paths):
        """
        Query GPT-4 Vision to rank a set of Fourier spectrum images
        
        Args:
            image_dir: Folder that contains Fourier spectrum images
        
        Returns:
            Ranked list of dictionaries with image paths and explanations
        """
        # Set up API credentials
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")


        # Prepare all images
        images_content = []
        for i, img_path in enumerate(image_paths):
            with open(img_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
            # Add image with label
            images_content.append({
                "type": "input_text",
                "text": f"Image {i}:"
            })
            images_content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            })
        
        # System prompt
        system_prompt = """
        You are an expert in image restoration quality assessment.
        You will be analyzing Fourier power spectra of several restored images.
        Rank them from best to worst quality based on:
        1. Natural frequency distribution
        2. Absence of artifacts (ringing, blocking)
        3. Noise characteristics
        4. Detail preservation
        
        Return a JSON object with the following structure:
        {
            "rankings": [
                {
                    "rank": 1,
                    "image_number": X
                },
                ...
            ]
        }
        
        Where rank 1 is the best quality image.
        """
        
        # User instruction
        user_instruction = {
            "type": "input_text",
            "text": f"Analyze these {len(image_paths)} Fourier power spectra and rank them from best (1) to worst quality. Return your response as a JSON object with the 'rankings' field containing the ordered list."
        }
        
        client = OpenAI()

        response = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_instruction["text"],
                }, 
                {
                    "role": "user",
                    "content": images_content
                }
            ],
        )

        try:
            parsed_content = json.loads(response.output[0].content[0].text.strip().removeprefix('```json\n').removesuffix('\n```'))
            
            # Validate the expected fields are present
            if "rankings" not in parsed_content:
                raise ValueError("Response missing required field 'rankings'")
            
            # Convert image numbers to actual paths for convenience
            for ranking in parsed_content["rankings"]:
                img_num = ranking["image_number"]
                ranking["image_path"] = image_paths[img_num]
            
            return parsed_content["rankings"]
        
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise ValueError(f"Failed to parse response: {str(e)}")
        
    def evaluate_driver(self, image_dir):
        files = os.listdir(image_dir)
        image_paths = [
            os.path.join(image_dir, file)
            for file in files
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))
        ]

        all_rankings = []
        for agent_id in range(self.n_agents):
            try:
                rankings = self.query_gpt_vision_for_ranking(image_paths)
                all_rankings.append(rankings)
                print(f"Agent {agent_id+1} rankings completed")
            except Exception as e:
                print(f"Error with Agent {agent_id+1}: {str(e)}")

        avg_ranks = {img_path: 0 for img_path in image_paths}
        rank_counts = {img_path: 0 for img_path in image_paths}

        for agent_rankings in all_rankings:
            for item in agent_rankings:
                img_path = item["image_path"]
                avg_ranks[img_path] += item["rank"]
                rank_counts[img_path] += 1
        
        # Calculate average rank
        for img_path in avg_ranks:
            if rank_counts[img_path] > 0:
                avg_ranks[img_path] = avg_ranks[img_path] / rank_counts[img_path]
        
        final_rankings = [
            {"image_path": img_path, "average_rank": avg_rank}
            for img_path, avg_rank in avg_ranks.items()
        ]
        
        # Sort by average rank (lowest is best)
        final_rankings.sort(key=lambda x: x["average_rank"])
        
        return final_rankings
            

if __name__ == "__main__":
    Avg = AverageEnsemble(n_agents=3)


    #Avg.query_gpt_vision_for_ranking('/Users/justin/Desktop/Everything/Code/agentic_images/files/spectra/closer')
    final_rankings = Avg.evaluate_driver('/Users/justin/Desktop/Everything/Code/agentic_images/files/spectra/closer')
    breakpoint()
    print("Does it work?")