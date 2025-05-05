# %%
import streamlit as st
import asyncio
import time
from typing import List, Callable
from openai import AsyncOpenAI
import os
from pydantic import BaseModel

# Initialize OpenAI client


# %%


s = "asdoih$dfsdoi"

s.replace("$", "\\$")
# %%
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class AnalogyList(BaseModel):
    class Analogy(BaseModel):
        excerpt: str
        visual_description: str
    analogies: list[Analogy]

response = await client.responses.parse(
    model="gpt-4.1-mini-2025-04-14",
    input=[
        {"role": "system", "content": "Extract passages (0-3) from the text that contain an analogy or a metaphor that can be visually represented. Repeat exactly the text from the input. Associate a precise visual representation to the passage."},
        {
            "role": "user",
            "content": "Hello",
        },
    ],
    text_format=AnalogyList
)

# %%


class TaskResponse(BaseModel):
    content: str
    completion_time: float

async def slow_function(name: str, complexity: int) -> TaskResponse:
    """A function that makes an async OpenAI call with varying complexity."""
    start_time = time.time()
    
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Write a {complexity}-sentence story about a magical {name}."}
            ],
            max_tokens=100
        )
        # get usage
        response.
        
        completion_time = time.time() - start_time
        return TaskResponse(
            content=f"Task {name} completed in {completion_time:.2f} seconds:\n{response.choices[0].message.content}",
            completion_time=completion_time
        )
    except Exception as e:
        completion_time = time.time() - start_time
        return TaskResponse(
            content=f"Task {name} failed after {completion_time:.2f} seconds: {str(e)}",
            completion_time=completion_time
        )

async def run_tasks_in_parallel(tasks: List[Callable]) -> List[TaskResponse]:
    """Run multiple async tasks in parallel with proper error handling."""
    try:
        return await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print(f"Error in run_tasks_in_parallel: {str(e)}")
        return []

def main():
    st.title("Async OpenAI Task Demo")
    
    # Add controls to sidebar
    with st.sidebar:
        num_tasks = st.slider("Number of tasks", min_value=1, max_value=30, value=3)
        complexity = st.slider("Story complexity (sentences)", min_value=1, max_value=5, value=2)
    
    if st.button("Run Tasks"):
        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create tasks with different names
        tasks = []
        for i in range(num_tasks):
            task_name = f"Story {i+1}"
            tasks.append(slow_function(task_name, complexity))
        
        # Update status
        status_text.text("Running OpenAI tasks in parallel...")
        
        # Run tasks and get results
        results = asyncio.run(run_tasks_in_parallel(tasks))
        
        # Update UI
        progress_bar.progress(100)
        status_text.text("All tasks completed!")
        
        # Display results
        st.subheader("Results:")
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                st.error(f"Task {i+1} failed: {str(result)}")
            else:
                st.write(result.content)
                st.write(f"Completion time: {result.completion_time:.2f} seconds")
                st.write("---")

if __name__ == "__main__":
    main() 