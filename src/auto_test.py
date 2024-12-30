from agent import Agent
from dataset import Dataset 

class AutoTest:
    def __init__(self, agent: Agent, dataset: Dataset):
        """
        :param agent: An Agent instance that implements .run(user_query) -> float
        :param dataset: A Dataset containing TestQuestions
        """
        self.agent = agent
        self.dataset = dataset

    def run_tests(self):
        """
        Iterate through all questions in the dataset, run the agent, 
        check correctness, and return results.
        Also prints the model's overall score as a percentage.
        """
        questions = self.dataset.questions
        results = {}
        correct_count = 0

        for tq in questions:
            model_answer = self.agent.run(tq.user_query)
            error = abs(model_answer - tq.answer)
            if error < tq.precision:
                results[tq] = 1
                correct_count += 1
            else:
                results[tq] = 0

        score = correct_count / len(questions) if questions else 0
        print(f"The model scored {score * 100:.1f}% on this testing set.")
        return results

if __name__ == '__main__':
    pass 