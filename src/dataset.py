import random
from typing import List, Literal 
from dataclasses import dataclass
from collections import Counter
from typing import Literal

@dataclass
class TestQuestion:
    difficulty: Literal["easy", "medium", "hard", "extremely hard"]
    user_query: str              # The SQL-based analytics question
    answer: float                # The numeric answer (for now, = -100)
    precision: float             # Tolerance for correctness

class Dataset:
    """
    A collection of TestQuestions.
    """
    def __init__(self, questions: List[TestQuestion]):
        self.questions = questions
        self.length    = len(questions)

    def sample(self, n: int) -> List[TestQuestion]:
        """
        Randomly sample n questions from the dataset.
        Raises ValueError if n > length of the dataset.
        """
        if n > self.length:
            raise ValueError(
                f"Cannot sample {n} questions from a dataset of length {self.length}."
            )
        return random.sample(self.questions, n)

    def print_info(self):
        """
        Print a nicely formatted report on:
          1) total questions
          2) number of questions by difficulty
          3) a sample question (i.e., user_query) from each difficulty that exists
        """
        print("=== Dataset Info ===")
        print(f"Total questions: {self.length}")

        # Count how many questions per difficulty
        difficulties = [q.difficulty for q in self.questions]
        diff_counts = Counter(difficulties)
        for diff, count in diff_counts.items():
            print(f"  {diff}: {count} question(s)")

        # For each difficulty, show one random example if it exists
        unique_difficulties = list(diff_counts.keys())
        print("\nSample questions by difficulty:")
        for diff in unique_difficulties:
            # Filter questions with difficulty == diff
            diff_questions = [q for q in self.questions if q.difficulty == diff]
            if diff_questions:
                example = random.choice(diff_questions)
                print(f"  - {diff.upper()} sample: '{example.user_query}'")
        print("====================")

if __name__ == '__main__':
    pass 