import io
from contextlib import redirect_stdout
import re
from src.chat import direct_chat
from src.utils import Instruction


class JudgeInstruction():
    @staticmethod
    def validator(x: str) -> bool:
        lines = x.strip().split("\n")
        last_line = lines[-1]
        
        # Check for plain text format in the last line
        if "Correct" in last_line or "Wrong" in last_line:
            return True
        
        # Check for LaTeX formats in the last few lines
        # Take the last 3 lines to handle multi-line LaTeX format
        last_few_lines = "\n".join(lines[-3:]) if len(lines) >= 3 else "\n".join(lines)
        
        # Pattern to match \[ \boxed{Correct/Wrong} \] across multiple lines
        boxed_pattern = r'\\?\[\s*\\boxed\{(Correct|Wrong)\}\s*\\?\]'
        if re.search(boxed_pattern, last_few_lines, re.MULTILINE | re.DOTALL):
            return True
        
        # Pattern to match \[ \boxed{\text{Correct/Wrong}} \] across multiple lines
        boxed_text_pattern = r'\\?\[\s*\\boxed\{\\text\{(Correct|Wrong)\}\}\s*\\?\]'
        if re.search(boxed_text_pattern, last_few_lines, re.MULTILINE | re.DOTALL):
            return True
        
        # Pattern to match \[ \text{Correct/Wrong} \] across multiple lines
        text_pattern = r'\\?\[\s*\\text\{(Correct|Wrong)\}\s*\\?\]'
        if re.search(text_pattern, last_few_lines, re.MULTILINE | re.DOTALL):
            return True
            
        # Also check for just \boxed{Correct/Wrong} without the brackets
        simple_boxed_pattern = r'\\boxed\{(Correct|Wrong)\}'
        if re.search(simple_boxed_pattern, last_few_lines):
            return True
        
        # Check for \boxed{\text{Correct/Wrong}} without the brackets
        simple_boxed_text_pattern = r'\\boxed\{\\text\{(Correct|Wrong)\}\}'
        return bool(re.search(simple_boxed_text_pattern, last_few_lines))

    @staticmethod
    def formatter(x: str) -> dict[str, str]:
        lines = x.strip().split("\n")
        last_line = lines[-1]

        if "Correct" in last_line or "Wrong" in last_line:
            # If the last line is plain text format
            judge = "Correct" if "Correct" in last_line else "Wrong"
            answer = "\n".join(lines[:-1])
            return {"answer": answer, "judge": judge}

        # Take the last 3 lines to handle multi-line LaTeX format
        last_few_lines = "\n".join(lines[-3:]) if len(lines) >= 3 else "\n".join(lines)
        
        # Try to extract from multi-line LaTeX boxed format: \[ \boxed{Correct/Wrong} \]
        boxed_pattern = r'\\?\[\s*\\boxed\{(Correct|Wrong)\}\s*\\?\]'
        boxed_match = re.search(boxed_pattern, last_few_lines, re.MULTILINE | re.DOTALL)
        
        if boxed_match:
            judge = boxed_match.group(1)
            latex_lines_count = JudgeInstruction._count_latex_lines(lines, ['\\[', '\\boxed', '\\]'])
            answer = "\n".join(lines[:-latex_lines_count]) if latex_lines_count > 0 else "\n".join(lines[:-1])
            return {"answer": answer, "judge": judge}
        
        # Try to extract from multi-line LaTeX boxed text format: \[ \boxed{\text{Correct/Wrong}} \]
        boxed_text_pattern = r'\\?\[\s*\\boxed\{\\text\{(Correct|Wrong)\}\}\s*\\?\]'
        boxed_text_match = re.search(boxed_text_pattern, last_few_lines, re.MULTILINE | re.DOTALL)
        
        if boxed_text_match:
            judge = boxed_text_match.group(1)
            latex_lines_count = JudgeInstruction._count_latex_lines(lines, ['\\[', '\\boxed', '\\text', '\\]'])
            answer = "\n".join(lines[:-latex_lines_count]) if latex_lines_count > 0 else "\n".join(lines[:-1])
            return {"answer": answer, "judge": judge}
        
        # Try to extract from multi-line LaTeX text format: \[ \text{Correct/Wrong} \]
        text_pattern = r'\\?\[\s*\\text\{(Correct|Wrong)\}\s*\\?\]'
        text_match = re.search(text_pattern, last_few_lines, re.MULTILINE | re.DOTALL)
        
        if text_match:
            judge = text_match.group(1)
            latex_lines_count = JudgeInstruction._count_latex_lines(lines, ['\\[', '\\text', '\\]'])
            answer = "\n".join(lines[:-latex_lines_count]) if latex_lines_count > 0 else "\n".join(lines[:-1])
            return {"answer": answer, "judge": judge}
        
        # Try single-line LaTeX boxed format
        simple_boxed_pattern = r'\\boxed\{(Correct|Wrong)\}'
        simple_match = re.search(simple_boxed_pattern, last_few_lines)
        
        if simple_match:
            judge = simple_match.group(1)
            # Remove the line containing \boxed
            for i, line in enumerate(reversed(lines)):
                if '\\boxed' in line:
                    answer = "\n".join(lines[:len(lines) - 1 - i])
                    break
            else:
                answer = "\n".join(lines[:-1])
            return {"answer": answer, "judge": judge}
        
        # Try single-line LaTeX boxed text format
        simple_boxed_text_pattern = r'\\boxed\{\\text\{(Correct|Wrong)\}\}'
        simple_text_match = re.search(simple_boxed_text_pattern, last_few_lines)
        
        if simple_text_match:
            judge = simple_text_match.group(1)
            # Remove the line containing \boxed{\text{...}}
            for i, line in enumerate(reversed(lines)):
                if '\\boxed' in line and '\\text' in line:
                    answer = "\n".join(lines[:len(lines) - 1 - i])
                    break
            else:
                answer = "\n".join(lines[:-1])
            return {"answer": answer, "judge": judge}
        
        # Fall back to plain text format
        judge = "Correct" if "Correct" in last_line else "Wrong"
        answer = "\n".join(lines[:-1])
        return {"answer": answer, "judge": judge}
    
    @staticmethod
    def _count_latex_lines(lines, keywords):
        """Count how many lines from the end contain LaTeX keywords"""
        latex_lines_count = 0
        for i in range(min(3, len(lines))):
            line_idx = len(lines) - 1 - i
            if any(keyword in lines[line_idx] for keyword in keywords):
                latex_lines_count = i + 1
            else:
                break
        return latex_lines_count
