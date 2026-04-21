import sys
import os

def main():
    print("=========================================================")
    print("🤖 Booting up Safe AutoML Tutor... (Loading ML libraries, this might take 30-60s on first run)")
    print("=========================================================\n")
    
    # Add solution_repo to sys.path so it can find local modules like guardrails
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solution_repo"))
    from agent import AutoMLAgent
    
    print("[System] ML Libraries Loaded!")
    print("Type 'exit' or 'quit' to end the chat.")
    print("[System] Ready! Say hello to your AI tutor. Remember to give it the dataset path like 'data/real_reviews_clean.csv' in your prompt.")
    
    agent = AutoMLAgent()

    while True:
        try:
            user_prompt = input("\n🧑 You: ").strip()
            if user_prompt.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
                
            if not user_prompt:
                continue
                
            print(f"🤖 Agent is thinking... (this may take a moment depending on the model)")
            
            # Send context to the agent and receive the response
            response = agent.process_request(user_prompt=user_prompt)
            
            print(f"\n🤖 Tutor:\n{response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()
