from create_embeding import read_source_documents, create_embedding

text = read_source_documents('https://docs.google.com/document/d/1gCvcpAgRrVjON801fBgwPBYyo46b3xT41bvxR_hN_b4/edit')
create_embedding(text)