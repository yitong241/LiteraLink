from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def summarize_text(text):
    # base_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # model = PeftModelForCausalLM.from_pretrained(base_model, "logits/llama2-7b-book-qa")
    # Load the summarization pipeline
    summarizer = pipeline("summarization", model='sshleifer/distilbart-cnn-12-6')
    # Generate summary
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Example usage
example_paragraph = """
Mr. Jones, of the Manor Farm, had locked the hen-houses for
the night, but was too drunk to remember to shut the pop-
holes. With the ring of light from his lantern dancing from
side to side, he lurched across the yard, kicked off his boots at the back
door, drew himself a last glass of beer from the barrel in the scullery,
and made his way up to bed, where Mrs. Jones was already snoring.
As soon as the light in the bedroom went out there was a stirring
and a fluttering all through the farm buildings. Word had gone round
during the day that old Major, the prize Middle White boar, had had a
strange dream on the previous night and wished to communicate it to
the other animals. It had been agreed that they should all meet in the
big barn as soon as Mr. Jones was safely out of the way. Old Major (so
he was always called, though the name under which he had been
exhibited was Willingdon Beauty) was so highly regarded on the farm
that everyone was quite ready to lose an hour’s sleep in order to hear
what he had to say.
At one end of the big barn, on a sort of raised platform, Major was
already ensconced on his bed of straw, under a lantern which hung
from a beam. He was twelve years old and had lately grown rather
stout, but he was still a majestic-looking pig, with a wise and
benevolent appearance in spite of the fact that his tushes had never
been cut. Before long the other animals began to arrive and make
themselves comfortable after their different fashions. First came the
three dogs, Bluebell, Jessie, and Pincher, and then the pigs, who
settled down in the straw immediately in front of the platform. The
hens perched themselves on the window-sills, the pigeons fluttered up
to the rafters, the sheep and cows lay down behind the pigs and began
to chew the cud. The two cart-horses, Boxer and Clover, came in
"""

# Generate summary
if __name__ == '__main__':
    summarized_text = summarize_text(example_paragraph)
    print(summarized_text)

