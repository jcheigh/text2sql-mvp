from dataset import Dataset, TestQuestion

def create_dataset():
    test_questions = [
        # ----------------- EASY -----------------
        TestQuestion(
            difficulty="easy",
            user_query="What is the average of all the values in the `open` column of the `ohlc` table?",
            answer=97.75121531858875,  
            precision=0.02
        ),
        TestQuestion(
            difficulty="easy",
            user_query="Find the smallest USD-to-EUR exchange rate stored in the `fxrates` table.",
            answer=0.8001354125807449,
            precision=1e-5
        ),
        TestQuestion(
            difficulty="easy",
            user_query="Among all rows in the `treasury_yields` table, what is the highest 10-year yield value?",
            answer=4.499022247446833,
            precision=1e-5
        ),

        # ----------------- MEDIUM -----------------
        TestQuestion(
            difficulty="medium",
            user_query="Compute the standard deviation of the difference between `close` and `open` in the `ohlc` table.",
            answer=17.65921497142543,
            precision=0.02
        ),
        TestQuestion(
            difficulty="medium",
            user_query="On average, how much higher is the 10-year yield than the 5-year yield in the `treasury_yields` table?",
            answer=0.32749703868504004,
            precision=0.02
        ),
        TestQuestion(
            difficulty="medium",
            user_query="Please tell me the correlation between `usd_to_eur` and `usd_to_gbp` in the `fxrates` table.",
            answer=-0.02545843934708713,
            precision=0.02
        ),

        # ----------------- HARD -----------------
        TestQuestion(
            difficulty="hard",
            user_query=(
                "Find the correlation between the stock price movement (`close - open`) in `ohlc` "
                "and the difference (`yield_7_year - yield_5_year`) in `treasury_yields`, matching rows by date."
            ),
            answer=-100,
            precision=0.001
        ),
        TestQuestion(
            difficulty="hard",
            user_query=(
                "Take the sum of `usd_to_eur` across all rows in `fxrates` and divide it by the sum of `usd_to_gbp` "
                "across all rows, then give me that ratio."
            ),
            answer=-100,
            precision=0.0001
        ),
        TestQuestion(
            difficulty="hard",
            user_query=(
                "Which day in `ohlc` has the largest range (`high - low`), and what is the 5-year Treasury yield "
                "on that same day? Please give me just that 5-year yield."
            ),
            answer=-100,
            precision=0.001
        ),

        # ----------------- EXTREMELY HARD -----------------
        TestQuestion(
            difficulty="extremely hard",
            user_query=(
                "Focus on the `ohlc` table. First, calculate the 75th percentile of `(close - open)`. "
                "Then filter rows where `(close - open)` is above that 75th percentile. "
                "For those filtered days, find the average 10-year yield (in `treasury_yields`)."
            ),
            answer=-100,
            precision=0.001
        ),
        TestQuestion(
            difficulty="extremely hard",
            user_query=(
                "Among all rows where the 7-year Treasury yield is higher than the 5-year yield, compute the correlation "
                "between `(close - open)` in `ohlc` and `usd_to_eur` in `fxrates`, matching rows by date."
            ),
            answer=-100,
            precision=0.001
        ),
        TestQuestion(
            difficulty="extremely hard",
            user_query=(
                "In `ohlc`, compute the mean and standard deviation of `close`. Filter to the days where `close` "
                "is more than 1.5 standard deviations above the mean. Then sum up `(yield_10_year - yield_5_year)` "
                "in `treasury_yields` for those days and give me the result."
            ),
            answer=-100,
            precision=0.01
        ),
    ]
    dataset = Dataset(test_questions[:6])
    dataset.print_info()
    return dataset