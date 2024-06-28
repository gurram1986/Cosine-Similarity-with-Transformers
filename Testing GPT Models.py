#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


# def calculate_cosine_similarity(array1, array2):
#     # Combine all sentences into a single list
#     sentences = array1 + array2
    
#     # Initialize CountVectorizer to convert sentences to vectors
#     vectorizer = CountVectorizer().fit_transform(sentences)
    
#     # Compute cosine similarity between the first len(array1) vectors and the next len(array2) vectors
#     cosine_similarities = cosine_similarity(vectorizer[:len(array1)], vectorizer[len(array1):])
    
#     return cosine_similarities


# In[6]:


def calculate_cosine_similarity(array1, array2):
    # Initialize CountVectorizer to convert sentences to vectors
    vectorizer = CountVectorizer().fit(array1 + array2)
    
    # Transform each array into vectors
    vectors1 = vectorizer.transform(array1)
    vectors2 = vectorizer.transform(array2)
    
    # Calculate cosine similarity for corresponding pairs
    similarities = []
    for vec1, vec2 in zip(vectors1, vectors2):
        similarity = cosine_similarity(vec1, vec2)
        similarities.append(similarity[0][0])  # Append the similarity score
        
    return similarities


# In[12]:


# Example usage:
if __name__ == "__main__":
    # Example arrays with sentences
    array1 = [
"Clouds form when warm air rises, cools down, and the water vapor in the air changes into tiny water droplets or ice crystals. These droplets or crystals come together to create a visible cloud. It's like when you see your breath on a cold day—it turns into tiny drops of water that you can see!",
"Zebra crossing lines are on roads to help people cross the street safely. The stripes make it clear to drivers that they need to slow down or stop for people walking. It's like a special path for walkers to get across the road.",
"One of the most popular kings in India is Akbar the Great. He was known for his fairness, strong leadership, and efforts to bring together people of different religions.",
"Fundamental rights for a USA citizen include freedoms protected by the Constitution, such as freedom of speech, religion, and the right to a fair trial. These rights are outlined primarily in the Bill of Rights and are essential for ensuring individual liberties and justice.",
"Border conflicts arise due to disputes over territorial claims, where countries contest the ownership of specific regions. They can also be fueled by ethnic, religious, or cultural differences that transcend national boundaries. Additionally, competition for resources like water, minerals, or oil can intensify these conflicts, as nations vie for control over valuable assets.",
"The human heart works like a pump that moves blood around your body. It has four parts called chambers that squeeze and relax to push blood through, giving your body the oxygen and nutrients it needs. Think of it like a super important engine that keeps everything running smoothly!",
"Global warming is primarily caused by the increased levels of greenhouse gases, such as carbon dioxide and methane, in the atmosphere due to human activities like burning fossil fuels and deforestation. These gases trap heat from the sun, causing the Earth's temperature to rise. Additionally, industrial activities and agricultural practices contribute to this effect, leading to changes in climate and weather patterns.",
"Shakespeare was an English playwright and poet who wrote famous plays like Romeo and Juliet and Hamlet. His works are celebrated worldwide for their themes of love, tragedy, and human nature.",
"Inflation in an economy can occur due to increased demand for goods and services compared to their supply (demand-pull inflation) or when production costs rise, leading to higher prices (cost-push inflation). Both factors can contribute to a general increase in prices across the economy over time.",
"Antibiotics work by targeting bacteria in the body, either by inhibiting their growth or by killing them outright. They do this through various mechanisms, such as interfering with bacterial cell wall synthesis, protein production, or DNA replication. Effective use of antibiotics requires matching the right antibiotic to the specific type of bacteria causing the infection."]
    
    array2 = [
"Clouds form when warm air rises and cools down. As the air cools, water vapor condenses into tiny droplets or ice crystals. These droplets or crystals gather together to form clouds, which we see in the sky.",
"Zebra crossing lines are painted on roads to help pedestrians safely cross the road. The black and white stripes make them easily visible, and drivers are required to stop and give way to pedestrians on these lines. It's a way to ensure everyone's safety while crossing the road.",
"India is a democratic country, and we do not have a king. However, we have had many popular leaders like Mahatma Gandhi, Jawaharlal Nehru, and Sardar Vallabhbhai Patel who played significant roles in shaping India's history and development.",
"Fundamental rights are the basic rights that are guaranteed to every citizen of the United States by the Constitution. These rights include freedom of speech, religion, press, and assembly, the right to bear arms, the right to a fair trial, and protection from unreasonable searches and seizures.",
"Border conflicts in the world can arise due to various reasons. Historical disputes over territorial claims, resource distribution, and ethnic or cultural differences can contribute to tensions between neighboring countries. Additionally, geopolitical factors, such as strategic interests, political instability, and competition for economic opportunities, can also fuel border conflicts.",
"The human heart is like a pump that keeps our body running. It receives oxygen-rich blood from the lungs and pumps it to the rest of the body through blood vessels. At the same time, it collects oxygen-poor blood and sends it back to the lungs to get refreshed.",
"Global warming is primarily caused by the increase in greenhouse gases, such as carbon dioxide, in the Earth's atmosphere. These gases trap heat from the sun, leading to a rise in temperature. Human activities like burning fossil fuels, deforestation, and industrial processes contribute to the excessive release of these gases, intensifying the greenhouse effect and causing global warming.",
"William Shakespeare was an influential English playwright and poet who lived in the 16th century. His timeless works, such as Romeo and Juliet and Hamlet, continue to be celebrated for their rich storytelling and profound insights into human nature.",
"Inflation occurs when there's a general increase in prices and a decrease in the purchasing power of money. Common causes include increased demand for goods and services, rising production costs, excessive money supply growth, and government policies that affect the economy.",
"Antibiotic medicine works by targeting and inhibiting the growth of bacteria in the human body. It does this by either killing the bacteria directly or preventing them from multiplying. This helps to control and eliminate bacterial infections, allowing the body's immune system to recover and heal."
    ]


# In[13]:


# Calculate cosine similarity
similarities = calculate_cosine_similarity(array1, array2)

# Print the similarity matrix
print("Cosine Similarity Matrix:")
print(similarities)


# In[ ]:




