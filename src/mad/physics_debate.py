"""
Physics Multi-Agent Debate Framework with Configuration Support
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

# Import the unified chat interface
from src.chat import direct_chat
from src.utils import MessageList


MODEL_DEFAULT = "deepseek-v3-250324"


class PhysicsDebateAgent:
    """A simple agent for physics debate"""
    
    def __init__(self, model_name: str, name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.conversation_history = []
    
    def set_system_prompt(self, prompt: str):
        """Set system prompt for the agent"""
        self.conversation_history = [{"role": "system", "content": prompt}]
    
    def add_message(self, content: str, role: str = "user"):
        """Add a message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
    
    async def query(self, message: str = None, record_path: str = None) -> str:
        """Query the model and get response using direct_chat"""
        if message:
            self.add_message(message, "user")
        
        try:
            # Use the unified direct_chat interface
            response = await direct_chat(
                model=self.model_name,
                messages=MessageList(self.conversation_history),
                record_full_api_path=record_path,
                temperature=self.temperature,
                max_retries=3
            )
            
            self.add_message(response, "assistant")
            return response
        
        except Exception as e:
            print(f"Error querying {self.name}: {e}")
            return f"Error: Unable to get response from {self.name}"


class PhysicsDebate:
    """Physics debate orchestrator with configuration support"""
    
    def __init__(self, 
                 config_file: str = "config.json",
                 debate_model: str = "deepseek-v3-250324",
                 temperature: float = 0.7,
                 max_rounds: int = 3):
        
        self.debate_model = debate_model
        self.temperature = temperature
        self.max_rounds = max_rounds
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize agents
        self.challenger = PhysicsDebateAgent(debate_model, "Challenger", temperature)
        self.defender = PhysicsDebateAgent(debate_model, "Defender", temperature)
        self.judge = PhysicsDebateAgent(debate_model, "Judge", temperature * 0.5)
        
        self.debate_log = []
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            # config_path = os.path.join(os.path.dirname(__file__), config_file)
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Fix LaTeX braces that conflict with Python string formatting
                for key, value in config.items():
                    if isinstance(value, str):
                        # Replace LaTeX boxed notation to avoid format conflicts
                        value = value.replace('\\[\\boxed{}\\]', '\\[\\boxed{{}}\\]')
                        value = value.replace('\\[ \\boxed{ final_answer} \\]', '\\[ \\boxed{{ final_answer}} \\]')
                        config[key] = value
                return config
        except FileNotFoundError:
            print(f"Warning: Config file {config_file} not found. Using default settings.")
            return {}
    
    def _setup_agents(self, question: str, reference_answer: str):
        """Setup agents using configuration templates"""

        challenger_prompt = self.config.get(
            "challenger_prompt_template",
            "You are a challenger in a physics debate."
        ).format(question=question, reference_answer=reference_answer)

        defender_prompt = self.config.get(
            "defender_prompt_template",
            "You are a defender in a physics debate."
        ).format(question=question, reference_answer=reference_answer)

        judge_prompt = self.config.get(
            "judge_prompt_template",
            "You are a judge in a physics debate."
        ).format(question=question, reference_answer=reference_answer)

        self.challenger.set_system_prompt(challenger_prompt)
        self.defender.set_system_prompt(defender_prompt)
        self.judge.set_system_prompt(judge_prompt)
    
    def _log_interaction(self, speaker: str, message: str, round_num: int = 0):
        """Log debate interactions"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "round": round_num,
            "speaker": speaker,
            "message": message
        }
        self.debate_log.append(entry)
        # print(f"\n--- {speaker} (Round {round_num}) ---")
        # print(message)
        # print("-" * 50)
    
    async def run_debate(self, task_id: str, question: str, reference_answer: str, output_aux_dir: str = None) -> Dict:
        """Run debate using configuration templates"""
        start_round = self._load_state(output_aux_dir)
        
        # Setup agents
        if start_round == 1:
            print(f"Starting new debate for task {task_id}...", flush=True)
            self._setup_agents(question, reference_answer)

        # Create record paths for API calls if output directory is provided
        record_base = None
        if output_aux_dir:
            os.makedirs(output_aux_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            record_base = os.path.join(output_aux_dir, f"debate_api_{timestamp}")
        
        # Get queries from config
        initial_query = self.config.get("challenger_initial_query", "Analyze the reference answer.")
        
        # Variables to track latest responses for follow-up rounds
        challenger_analysis = ""
        defender_response = ""
        
        # Round 1: Initial analysis (only if not resuming from later round)
        if start_round <= 1:
            challenger_record = f"{record_base}_challenger_r1.json" if record_base else None
            challenger_analysis = await self.challenger.query(initial_query, record_path=challenger_record)
            self._log_interaction("Challenger", challenger_analysis, 1)
            
            # Round 1: Defender response
            defender_query = self.config.get("defender_response_template", "Respond to the challenger.").format(
                challenger_analysis=challenger_analysis
            )
            defender_record = f"{record_base}_defender_r1.json" if record_base else None
            defender_response = await self.defender.query(defender_query, record_path=defender_record)
            self._log_interaction("Defender", defender_response, 1)
            
            # Save state after completing round 1
            self._save_state(output_aux_dir, 1)
        else:
            # Extract last responses from debate log for continuity
            for entry in reversed(self.debate_log):
                if entry['speaker'] == 'Defender' and defender_response == "":
                    defender_response = entry['message']
                elif entry['speaker'] == 'Challenger' and challenger_analysis == "":
                    challenger_analysis = entry['message']
                if challenger_analysis and defender_response:
                    break
        
        # Additional rounds (start from appropriate round)
        for round_num in range(max(2, start_round), self.max_rounds + 1):
            # Challenger follow-up
            challenger_query = self.config.get("challenger_followup_template", "Provide follow-up.").format(
                defender_response=defender_response
            )
            challenger_record = f"{record_base}_challenger_r{round_num}.json" if record_base else None
            challenger_followup = await self.challenger.query(challenger_query, record_path=challenger_record)
            self._log_interaction("Challenger", challenger_followup, round_num)
            
            # Defender follow-up  
            defender_query = self.config.get("defender_followup_template", "Respond to follow-up.").format(
                challenger_followup=challenger_followup
            )
            defender_record = f"{record_base}_defender_r{round_num}.json" if record_base else None
            defender_response = await self.defender.query(defender_query, record_path=defender_record)
            self._log_interaction("Defender", defender_response, round_num)
            
            # Save state after completing each round
            self._save_state(output_aux_dir, round_num)
        
        # Judge's decision
        debate_summary = "\n\n".join([
            f"{entry['speaker']}: {entry['message']}" 
            for entry in self.debate_log[-4:]
        ])
        
        judge_query = self.config.get("judge_final_query_template", "Make final judgment.").format(
            debate_summary=debate_summary
        )
        judge_record = f"{record_base}_judge_final.json" if record_base else None
        judge_verdict = await self.judge.query(judge_query, record_path=judge_record)
        self._log_interaction("Judge", judge_verdict, "Final")
        
        # Save completion marker and clean up state file after successful completion
        result = {
            "original_question": question,
            "reference_answer": reference_answer,
            "final_verdict": judge_verdict,
            "debate_log": self.debate_log,
            "summary": self._extract_final_answer(judge_verdict)
        }
        
        if output_aux_dir:
            # Save completion marker with result
            self._save_completion_marker(output_aux_dir, task_id, result)
            
            # Clean up state file
            state_file = os.path.join(output_aux_dir, "debate_state.json")
            try:
                if os.path.exists(state_file):
                    os.remove(state_file)
            except Exception as e:
                print(f"Warning: Failed to clean up state file: {e}")
        
        return result
    
    def _save_state(self, output_aux_dir: str, completed_rounds: int):
        """Save current debate state to enable resumption from breakpoint"""
        if not output_aux_dir:
            print("Warning: No output directory specified, skipping state save.",
                  flush=True)
            return
        
        state_file = os.path.join(output_aux_dir, "debate_state.json")
        state_data = {
            "completed_rounds": completed_rounds,
            "challenger_history": self.challenger.conversation_history,
            "defender_history": self.defender.conversation_history,
            "judge_history": self.judge.conversation_history,
            "debate_log": self.debate_log,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save debate state: {e}",
                  flush=True)
    
    def _load_state(self, output_aux_dir: str) -> int:
        """Load debate state from previous session, return starting round number"""
        if not output_aux_dir:
            return 1
        
        state_file = os.path.join(output_aux_dir, "debate_state.json")
        if not os.path.exists(state_file):
            return 1
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # Restore agent conversation histories
            self.challenger.conversation_history = state_data.get("challenger_history", [])
            self.defender.conversation_history = state_data.get("defender_history", [])
            self.judge.conversation_history = state_data.get("judge_history", [])
            
            # Restore debate log
            self.debate_log = state_data.get("debate_log", [])
            
            completed_rounds = state_data.get("completed_rounds", 0)
            
            return completed_rounds + 1
            
        except Exception as e:
            print(f"Warning: Failed to load debate state, starting fresh: {e}",
                  flush=True)
            return 1
    
    def _save_completion_marker(self, output_aux_dir: str, task_id: str, result: Dict):
        """Save completion marker with final result for future quick loading"""
        if not output_aux_dir:
            print("Warning: No output directory specified, skipping completion marker save.",
                  flush=True)
            return
        
        completion_file = os.path.join(output_aux_dir, "debate_completed.json")
        completion_data = {
            "task_id": task_id,
            "completed": True,
            "completion_timestamp": datetime.now().isoformat(),
            "final_result": result,
            "metadata": {
                "model": self.debate_model,
                "temperature": self.temperature,
                "max_rounds": self.max_rounds
            }
        }
        
        try:
            with open(completion_file, 'w', encoding='utf-8') as f:
                json.dump(completion_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Debate completion marker saved for task {task_id}", flush=True)
        except Exception as e:
            print(f"⚠️ Warning: Failed to save completion marker: {e}", flush=True)
    
    def _load_completion_marker(self, output_aux_dir: str, task_id: str) -> Optional[Dict]:
        """Load completion marker if debate was already completed"""
        if not output_aux_dir:
            return None
        
        completion_file = os.path.join(output_aux_dir, "debate_completed.json")
        if not os.path.exists(completion_file):
            return None
        
        try:
            with open(completion_file, 'r', encoding='utf-8') as f:
                completion_data = json.load(f)
            
            # Verify this is the same task
            if completion_data.get("task_id") == task_id and completion_data.get("completed"):
                print(f"✅ Found completed debate for task {task_id}, loading previous result...", flush=True)
                return completion_data.get("final_result")
            
        except Exception as e:
            print(f"Warning: Failed to load completion marker: {e}", flush=True)
        
        return None
    
    def _extract_final_answer(self, judge_verdict: str) -> str:
        """Extract final answer from judge verdict"""
        try:
            if "FINAL ANSWER:" in judge_verdict:
                parts = judge_verdict.split("FINAL ANSWER:")[1].split("JUSTIFICATION:")[0].strip()
                return parts
            else:
                return judge_verdict
        except Exception as e:
            return judge_verdict
    
    def save_debate_log(self, output_dir: str, filename: str = None) -> str:
        """Save detailed debate log with metadata"""
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"physics_debate_{timestamp}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        detailed_log = {
            "metadata": {
                "model": self.debate_model,
                "temperature": self.temperature,
                "max_rounds": self.max_rounds,
                "config_used": self.config != {}
            },
            "debate_log": self.debate_log
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(detailed_log, f, ensure_ascii=False, indent=2)
        
        return filepath


async def improve_answer(
    task_id: str,
    question: str, 
    reference_answer: str, 
    output_aux_dir: str,
    config_file: str = "config.json",
    debate_model: str = "deepseek-v3-250324",
    temperature: float = 0.0,
    max_rounds: int = 3,
    screen_print: bool = True
) -> str:
    """
    Improve a physics problem answer through multi-agent debate.
    
    Args:
        task_id: Unique identifier for the task
        question: Physics problem statement
        reference_answer: Reference answer to evaluate/improve
        output_aux_dir: Directory to save debate logs
        config_file: JSON configuration file path
        debate_model: LLM model name
        temperature: Sampling temperature
        max_rounds: Maximum debate rounds
        screen_print: Whether to print output to console
    
    Returns:
        str: Improved final answer
    """
    
    # Initialize debate
    debate = PhysicsDebate(
        config_file=config_file,
        debate_model=debate_model,
        temperature=temperature,
        max_rounds=max_rounds
    )
    
    # Run debate
    if not screen_print:
        print(f"Starting physics debate...")
        print(f"Question: {question}")
        print(f"Reference Answer: {reference_answer}")
        print("=" * 80)
    
    result = await debate.run_debate(task_id, question, reference_answer, output_aux_dir)
    
    # Save logs
    saved_path = debate.save_debate_log(output_aux_dir)
    
    print(f"\nDebate completed for {task_id}! Log saved to: {saved_path}")
    if not screen_print:
        print(f"Final Answer: {result['summary']}")
    
    return result['summary']


async def mad_solver(
    task_id: str,
    question_statement: str,
    solution: str,
    output_aux_path: Optional[str] = None,
    debate_model: Optional[str] = None,
    temperature: float = 0.0,
    max_rounds: int = 3,
    config_file: str = "config.json"
) -> tuple[bool, str]:
    """
    MAD solver: Multi-Agent Debate solver for physics problems.
    
    Args:
        task_id: The unique identifier for the current task
        question_statement: The physics question statement
        solution: The initial solution to debate and improve
        output_aux_path: Path of output aux directory for logs
        debate_model: The model to use for debate agents
        temperature: Temperature setting for the models
        max_rounds: Maximum number of debate rounds
        config_file: Configuration file for debate prompts
        
    Returns:
        tuple[bool, str]: (success, result_text)
            - If successful: (True, improved_solution_text)
            - If failed due to max rounds exceeded: (False, original_solution)
    """
    
    # Use default model if not specified
    if debate_model is None:
        debate_model = MODEL_DEFAULT
    
    try:
        # Initialize debate with configuration
        debate = PhysicsDebate(
            config_file=config_file,
            debate_model=debate_model,
            temperature=temperature,
            max_rounds=max_rounds
        )
        
        # Check if debate was already completed for this task
        if output_aux_path:
            completed_result = debate._load_completion_marker(output_aux_path, task_id)
            if completed_result:
                # Return the previously completed result
                improved_answer = completed_result.get('summary', '')
                if improved_answer and improved_answer.strip():
                    return True, improved_answer
                else:
                    return False, "Origin Answer Contradict"
        
        # Run the debate (will handle breakpoint resumption internally)
        result = await debate.run_debate(
            task_id=task_id,
            question=question_statement,
            reference_answer=solution,
            output_aux_dir=output_aux_path
        )
        
        # Save detailed logs if output path is provided
        if output_aux_path:
            debate.save_debate_log(output_aux_path, f"{task_id}_debate_log.json")
        
        # Extract the final improved answer
        improved_answer = result['summary']
        
        # Check if we have a meaningful improvement or if debate was inconclusive
        # For now, we consider the debate successful if we get any result
        # In the future, we could add more sophisticated success criteria
        if improved_answer and improved_answer.strip():
            return True, improved_answer
        else:
            # If no meaningful result, return False
            return False, "Origin Answer Contradict"
            
    except Exception as e:
        print(f"Error in MAD solver for task {task_id}: {e}")
        import traceback
        traceback.print_exc()
        # Return False
        return False, "Origin Answer Contradict"


if __name__ == "__main__":
    async def main():
        # Test the debate framework
        sample_question = """
        A 0.5 kg ball is dropped from a height of 10 m. When it hits the ground, 
        it bounces back to a height of 6 m. Calculate the coefficient of restitution.
        """
        
        sample_reference = """
        The coefficient of restitution e = sqrt(h2/h1) where h1 is initial height and h2 is bounce height.
        e = sqrt(6/10) = sqrt(0.6) = 0.775
        """
        
        try:
            # Test the new mad_solver function
            success, final_answer = await mad_solver(
                task_id="test_debate_001",
                question_statement=sample_question,
                solution=sample_reference,
                output_aux_path="./debate_logs",
                debate_model="deepseek-v3-250324",
                temperature=0.0,
                max_rounds=3
            )
            
            print(f"\nMAD Solver Result:")
            print(f"Success: {success}")
            print(f"Final answer: {final_answer}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Run the async main function
    asyncio.run(main())
