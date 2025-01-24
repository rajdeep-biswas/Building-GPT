def byte_pair_encode(tokens, depth = 0):

  # pair_mapping_dict will contain new_token: token_pair key value pairs {new_token: (token, token)} for pairs of token that occur more than once in a series of tokens
  pair_mapping_dict = dict()
  pair_mapping_dict_rev = dict()

  # it gets convoluted to just use pair_mapping_dict and list of tokens in present iteration to reliably check if a token is already in use
  # so we use a global set of all tokens ever used. also a find_new_token is initialized to count upwards until we find an unused token that can represent a pair of tokens
  set_of_used_tokens = set(tokens)
  find_new_token = 1

  depth_counter = 1

  while True:

    # setting depth to 0 would mean compression occurs until no more pairs occur more than once. this can be used as a hyperparameter to influence the vocabulary size
    if depth_counter == depth:
      break
    
    depth_counter += 1

    # a better way to choose the vocabulary size is to do the following. this will need an additional method parameter vocab_size. i am skipping the implementation for now since it's relatively intuitive and simple
    """
    if len(set_of_used_tokens) > vocab_size:
      return [...]
    """

    # first scan through the list of tokens to get pair frequencies
    pair_frequency_dict = dict()

    for i in range(len(tokens) - 1):
      byte_pair_tuple = (tokens[i], tokens[i + 1])
      pair_frequency_dict[byte_pair_tuple] = 1 + pair_frequency_dict.get(byte_pair_tuple, 0)

    # if we don't find pairs that occur more than once, we have encoded the list of tokens to the minimal possible
    if max(pair_frequency_dict.values()) <= 1:
      break


    # iteratively generate compressed list of tokens
    new_tokens = []

    i = 0
    while i < len(tokens) - 1:

      # pick up a pair of tokens
      byte_pair_tuple = (tokens[i], tokens[i + 1])

      # if said pair has occurred more than once
      if pair_frequency_dict[byte_pair_tuple] > 1:

        if byte_pair_tuple not in pair_mapping_dict_rev:

          # count up until we find an unused token
          while find_new_token in set_of_used_tokens:
            find_new_token += 1

          # add said token to the set that keeps track of tokens in use
          set_of_used_tokens.add(find_new_token)

          pair_mapping_dict_rev[byte_pair_tuple] = find_new_token
            
          # populate dictionary and list
          pair_mapping_dict[find_new_token] = byte_pair_tuple
          new_tokens.append(find_new_token)

        else:

          new_tokens.append(pair_mapping_dict_rev[byte_pair_tuple])

        # skip over two tokens since we have already compressed both, so it's erraneous to reuse the (i + 1)th token
        i += 2

      else:

        # if pair wasn't compressed, just proceed to look at the next token
        new_tokens.append(byte_pair_tuple[0])
        i += 1

    # we might have jumped over the last token from the i += 2 above, so we check and populate any leftover token
    if i == len(tokens) - 1:
      new_tokens.append(tokens[-1])

    # reassign the list for the next iteration
    tokens = new_tokens

  print("depth reached:", depth_counter, "vocab size:", len(set_of_used_tokens), "result token length:", len(tokens))

  # return encoded_tokens, pair mapping dictionary needed for decoding, depth reached / used, vocabulary size
  return tokens, pair_mapping_dict, depth_counter, len(set_of_used_tokens)


def byte_pair_decode(tokens, pair_mapping_dict):

  while True:

    # we want to keep iterating until all nested compressed tokens are resolved
    match_found = False

    # we want to compress backwards so we used descending order of new tokens generated
    for encoded_token in sorted(pair_mapping_dict.keys())[::-1]:

      original_tokens = []

      # check each token in the list that maybe a compressed form and use the reversed logic of byte_pair_encode
      for token in tokens:
        if token == encoded_token:
          original_tokens.extend(pair_mapping_dict[token])
          match_found = True
        else:
          original_tokens.append(token)

      tokens = original_tokens

      if not match_found:
        break

    return tokens

# testing

sample_string = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
tokens = sample_string.encode('utf-8')
tokens = list(map(int, tokens))

encoded_tokens, pair_mapping_dict = byte_pair_encode(tokens)

original_tokens = byte_pair_decode(encoded_tokens, pair_mapping_dict)

print(pair_mapping_dict)
print(encoded_tokens)
print(tokens)
print(original_tokens)
print(tokens == original_tokens)

from prettytable import PrettyTable
table = PrettyTable()

table.field_names = ["Depth", "Token Length", "Compression Ratio", "Vocabulary Size"]
table.add_row(["Original", len(tokens), len(tokens) / len(tokens), len(set(tokens))])

for i in range(1, 10):
  encoded_tokens, _, depth_counter, vocab_size = byte_pair_encode(tokens, depth = i)
  table.add_row([depth_counter, len(encoded_tokens), f"{len(tokens) / len(encoded_tokens):.2f}", vocab_size])

print(table)
