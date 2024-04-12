# dpq: data. prompt. query.

dpq is a Python library that makes it easy to process data and engineer features using
generative AI.

![dpq_demo](https://github.com/data-prompt-query/pre-release/assets/15915676/ea08d1ec-bf2d-473d-b521-d1ae9581050a)

## quick start
```python
import dpq

# Initialize dpq agent with API configuration
dpq_agent = dpq.Agent(
    url="ENDPOINT_URL",
    api_key="YOUR_API_KEY",
    model="MODEL_ID",
    custom_messages_path="OPTIONAL_PATH_TO_CUSTOM_PROMPTS"
)

# Apply prompt to each item in list-like iterable such as pandas series
dpq_agent.classify_sentiment(df['some_column'])
```

## adding functionalities
A function is defined by a `JSON` holding messages.

```
[
    {
        "role": "system",
        "content": "You are a sentiment classifier. You classify statements as having
         either a positive or negative sentiment. You return only one of two words:
         positive, negative."
    },
    {
        "role": "user",
        "content": "I like dpq. It makes prompt-based feature engineering a breeze."
    },
    {
        "role": "assistant",
        "content": "positive"
    }
]
```

To add a new function, simply add the `JSON` file to a prompts folder on your system and
initialize the dpq agent with the respective `custom_messages_path` pointing to the
folder. The function name is automatically set to the name of the `JSON` file.

Alternatively, you can pass the messages to generate a new function directly in your
code.

```python
# Define messages
messages = [
    {
        "role": "system",
        "content": "You return the country of a city."
    },
    {
        "role": "user",
        "content": "Berlin"
    },
    {
        "role": "assistant",
        "content": "Germany"
    },
]

# Add new function
dpq_agent.return_country = dpq_agent.generate_function(messages)

# Apply to a list
dpq_agent.return_country(["Berlin", "London", "Paris"])
```

## examples
In addition to the prompts in the `prompts` directory, which are loaded by default when
initializing the `dpq.Agent()`, we maintain a library of additional examples in the
`examples` directory. These are typically slightly less general-purpose. Feel free to
open a pull request and share prompts you have found useful with everyone!

## features
- feature engineering using prompts
- library of standard functions
- parallelized by default

## compatibility
dpq uses the `requests` library to send [OpenAI-style](https://platform.openai.com/docs/guides/text-generation/chat-completions-api)
Chat Completions API requests. For GPT-3.5 Turbo, the configuration is as follows.

```python
dpq_agent = dpq.Agent(
    url="https://api.openai.com/v1/chat/completions",
    api_key="YOUR_API_KEY",
    model="gpt-3.5-turbo",
)
```

## costs and speed
dpq currently comes as is without cost or speed guarantees. To still give a very rough
estimate: on a test data set of 1000 product reviews, the `classify_sentiment.json`
finishes in approx. 30 seconds (parallelized) on a standard Macbook and costs
$0.05 using `gpt-3.5-turbo`.

## is using LLMs a good idea?
Recent studies have shown promising results using general-purpose LLMs for text
annotation and classification. For example, [Gilardi, Alizadeh, and Kubli (2023)](https://doi.org/10.48550/arXiv.2303.15056)
and [Törnberg (2023)](https://doi.org/10.48550/arXiv.2304.06588) report
better-than-human performance. This is an active research area and we are looking
forward to seeing more results in this field. In general, we believe that LLMs can
deliver **consistent, high-quality output resulting in scalability, reduced time and
costs** (see also [Aguda (2024)](https://doi.org/10.48550/arXiv.2403.18152)).

