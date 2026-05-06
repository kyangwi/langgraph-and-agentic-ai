from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
import datetime

actor_prompt_template =ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        Tiy are an expert AI researcher.
        Current time: {time}
        1.{first_instruction}
        2. Reflect and critique your answer. Be severe to maximize improvement.
        3. After the reflection, **list 1-3 search queries separately** for researching improvements.
        Do not include them inside the reflection""",),
        MessagesPlaceholder(variable_name="messages"),
        ("system","Answer the user's question using required format")
    ]
)