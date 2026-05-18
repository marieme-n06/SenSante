import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERREUR : GROQ_API_KEY non trouvee dans .env")
    exit()

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system",
         "content": "Tu es un assistant medical senegalais. Reponds en francais simple. Maximum 3 phrases."},
        {"role": "user",
         "content": "Quels sont les symptomes du paludisme ?"}
    ],
    max_tokens=200,
    temperature=0.3
)

print("=== Reponse de Llama 3 ===")
print(response.choices[0].message.content)
print(f"\nTokens utilises : {response.usage.total_tokens}")
# Exercice 2 : tester differentes temperatures
for temp in [0.0, 0.5, 1.0]:
    print(f"\n=== Temperature LLM : {temp} ===")
    rep = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system",
             "content": "Tu es un assistant medical senegalais. Reponds en francais simple. Maximum 3 phrases."},
            {"role": "user",
             "content": "Patient : Homme, 35 ans, temperature 39.5C. Diagnostic : paludisme (72%). Explique."}
        ],
        max_tokens=200,
        temperature=temp
    )
    print(rep.choices[0].message.content)