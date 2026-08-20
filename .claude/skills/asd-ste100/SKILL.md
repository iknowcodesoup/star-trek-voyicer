---
name: asd-ste100
description: This skill provides guidance for writing plain, clear English in any output — chat replies, code comments, commit messages, PR descriptions, spec prose, and UI strings. Use this skill for any task that produces written text, not just code.
---

## Steps

### 1. Keep Sentences Short

Write short sentences. Use 20 words or fewer for instructions. Use 25 words or fewer for descriptions. Split long sentences into two short sentences.

| Bad                                                                                                                          | Good                                                       |
| ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| The system, which was built to process requests concurrently, should generally handle them without much delay in most cases. | The system handles requests concurrently. It has no delay. |

### 2. One Idea Per Sentence

Write one idea in each sentence. Do not join two actions with "and" or "then". Do not stack clauses with "which" or "in order to".

| Bad                                                                              | Good                                                                       |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Run the build, which will compile the code and then deploy it if the tests pass. | Run the build. It compiles the code. It deploys the code after tests pass. |

### 3. Use Active Voice and Commands

Use active voice. Use commands for instructions. Do not use passive voice unless the actor is unknown.

| Bad                                                | Good                 |
| -------------------------------------------------- | -------------------- |
| The button should be clicked by the user.          | Click the button.    |
| You might want to consider restarting the service. | Restart the service. |

### 4. Use Present Tense

Use present tense for facts and states. Use simple past tense for finished actions. Do not stack conditional tenses.

| Bad                                                                      | Good                                           |
| ------------------------------------------------------------------------ | ---------------------------------------------- |
| The process would have been able to complete if the disk had more space. | The process needs more disk space to complete. |

### 5. Use One Word for One Meaning

Pick one word for each idea. Use that word every time. Do not swap in a synonym for variety. Apply the same rule as `ENUMS NOT STRINGS` and `PATTERN NAMES` in Quick Reference, but to prose instead of code.

| Bad                                                                  | Good                                                        |
| -------------------------------------------------------------------- | ----------------------------------------------------------- |
| Delete the file, then remove the record, then erase the cache entry. | Delete the file. Delete the record. Delete the cache entry. |

### 6. Avoid Long Noun Clusters

Do not chain more than three nouns together. Rewrite noun clusters as a phrase with prepositions.

| Bad                                   | Good                                               |
| ------------------------------------- | -------------------------------------------------- |
| the user data retention policy engine | the engine that enforces the data-retention policy |

### 7. Avoid Jargon, Idioms, and Filler Words

Do not use jargon, idioms, or filler words. Do not hedge. State the fact.

| Bad                                                               | Good                            |
| ----------------------------------------------------------------- | ------------------------------- |
| Under the hood, we leverage a cache to facilitate faster lookups. | The cache makes lookups faster. |
| This could possibly cause an issue at some point.                 | This causes a bug.              |

### 8. Use Exact Numbers, Not Vague Quantifiers

Use exact counts when you know them. Do not use "some", "several", or "a few".

| Bad                   | Good            |
| --------------------- | --------------- |
| Several tests failed. | 3 tests failed. |

### 9. Use Numbered Lists for Steps

Use a numbered list for a sequence of steps. Do not describe a sequence in one paragraph.

| Bad                                                                    | Good                                                        |
| ---------------------------------------------------------------------- | ----------------------------------------------------------- |
| First build the project, then run the tests, and after that deploy it. | 1. Build the project.<br>2. Run the tests.<br>3. Deploy it. |

## Applies To

| Output Type     | Examples                                             |
| --------------- | ---------------------------------------------------- |
| Chat responses  | Explanations, status updates, summaries              |
| Code comments   | The rare comment that explains a non-obvious reason  |
| Commit messages | Subject line and body                                |
| PR descriptions | Summary and test plan                                |
| Spec prose      | Context, Goal, Design, and Notes sections            |
| UI copy         | React labels, error messages, dialog text            |
| API responses   | FastAPI `detail` strings, validation messages        |
| Log messages    | `logger.info`, `logger.warning`, `logger.error` text |
