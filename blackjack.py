from keras.models import model_from_json
import numpy as np
import pandas as pd
from blackjack_sim import *
from PyQt5 import *



def model_decision(model, player_sum, has_ace, dealer_card_num, new_stack,
                   games_played, card_count):
    input_array = np.array([player_sum, 0, has_ace,
                            dealer_card_num]).reshape(1, -1)
    cc_array = pd.DataFrame.from_dict([card_count])
    input_array = np.concatenate([input_array, cc_array], axis=1)
    predict_correct = model.predict(input_array)
    if predict_correct >= 0.52:
        print('hit', predict_correct)
        return 1, predict_correct
    else:
        print('stay', predict_correct)
        return 0, predict_correct


# ['player_total_initial', 'hit?', 'has_ace', 'dealer_card_num', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'A']

if __name__ == "__main__":
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

    card_count = {
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        'A': 0
    }

#
#     # Play blackjack but use the neural net to make hit/stay decision
# # And use a second neural net to decide how much to bet based on probability of dealer busting
#
# nights = 101
# bankrolls = []
#
# for night in range(nights):
#
#     dollars = 10000
#     bankroll = []
#     stacks = 101
#     players = 1
#     num_decks = 1
#
#     card_types = ['A',2,3,4,5,6,7,8,9,10,10,10,10]
#
#     dealer_card_feature = []
#     player_card_feature = []
#     player_live_total = []
#     player_live_action = []
#     player_results = []
#
#     first_game = True
#     prev_stack = 0
#     stack_num_list = []
#     new_stack = []
#     card_count_list = []
#     games_played_with_stack = []
#
#
#     for stack in range(stacks):
#         games_played = 0
#
#         if stack != prev_stack:
#             temp_new_stack = 1
#         else:
#             temp_new_stack = 0
#
#         # Make a dict for keeping track of the count for a stack
#         card_count = {2: 0,
#                       3: 0,
#                       4: 0,
#                       5: 0,
#                       6: 0,
#                       7: 0,
#                       8: 0,
#                       9: 0,
#                       10: 0,
#                       'A': 0}
#         human_count = 0
#
#         blackjack = set(['A',10])
#         dealer_cards = make_decks(num_decks, card_types)
#         while len(dealer_cards) > 20:
#             multiplier = 1
#
#             curr_player_results = np.zeros((1,players))
#
#             dealer_hand = []
#             player_hands = [[] for player in range(players)]
#             live_total = []
#             live_action = []
#
#             # Record card count
#             cc_array_bust = pd.DataFrame.from_dict([card_count])
#
#             # Deal FIRST card
#             for player, hand in enumerate(player_hands):
#                 card = dealer_cards.pop(0)
#                 player_hands[player].append(card)
#                 card_count[card] += 1
#                 if card in [10, 'A']:
#                     human_count -= 1
#                 elif card in [2, 3, 4, 5, 6]:
#                     human_count += 1
#             card = dealer_cards.pop(0)
#             dealer_hand.append(card)
#             card_count[card] += 1
#             if card in [10, 'A']:
#                 human_count -= 1
#             elif card in [2, 3, 4, 5, 6]:
#                 human_count += 1
#             # Deal SECOND card
#             for player, hand in enumerate(player_hands):
#                 card = dealer_cards.pop(0)
#                 player_hands[player].append(card)
#                 card_count[card] += 1
#                 if card in [10, 'A']:
#                     human_count -= 1
#                 elif card in [2, 3, 4, 5, 6]:
#                     human_count += 1
#             dealer_hand.append(dealer_cards.pop(0))
#
#             # Record the player's live total after cards are dealt
#             live_total.append(total_up(player_hands[player]))
#             action = 0
#             print("dealer hand: " + str(dealer_hand[0]) + "\nplayer hand: " + str(player_hands[player])
#                   + "\ncount: " + str(human_count))
#             # Dealer checks for 21. need to change
#             if set(dealer_hand) == blackjack:
#                 for player in range(players):
#                     if set(player_hands[player]) != blackjack:
#                         curr_player_results[0,player] = -1
#                     else:
#                         curr_player_results[0,player] = 0
#             else:
#                 for player in range(players):
#                     # Players check for 21
#                     if set(player_hands[player]) == blackjack:
#                         curr_player_results[0,player] = 1
#                         multiplier = 1.25
#                     else:
#                         # Neural net decides whether to hit or stay
#                         if 'A' in player_hands[player]:
#                             ace_in_hand = 1
#                         else:
#                             ace_in_hand = 0
#                         if dealer_hand[0] == 'A':
#                             dealer_face_up_card = 11
#                         else:
#                             dealer_face_up_card = dealer_hand[0]
#
#                         while (model_decision(model, total_up(player_hands[player]),
#                                                   ace_in_hand, dealer_face_up_card,
#                                                   temp_new_stack, games_played,
#                                                   card_count
#                                                   )[0] == 1) and (total_up(player_hands[player]) != 21):
#                             card = dealer_cards.pop(0)
#                             player_hands[player].append(card)
#                             card_count[card] += 1
#                             if card in [10, 'A']:
#                                 human_count -= 1
#                             elif card in [2, 3, 4, 5, 6]:
#                                 human_count += 1
#                             action = 1
#                             live_total.append(total_up(player_hands[player]))
#                             print("dealer hand: " + str(dealer_hand[0]) +  "\nplayer hand: " + str(player_hands[player])
#                                   + "\ncount: " + str(human_count))
#                             if total_up(player_hands[player]) > 21:
#                                 curr_player_results[0,player] = -1
#                                 print('bust')
#                                 break
#
#             # Dealer hits based on the rules
#             card_count[dealer_hand[-1]] += 1
#             while total_up(dealer_hand) < 17:
#                 card = dealer_cards.pop(0)
#                 dealer_hand.append(card)
#                 card_count[card] += 1
#                 if card in [10, 'A']:
#                     human_count -= 1
#                 elif card in [2, 3, 4, 5, 6]:
#                     human_count += 1
#             print("dealer total: " + str(dealer_hand) )
#             # Compare dealer hand to players hand but first check if dealer busted
#             if total_up(dealer_hand) > 21:
#                 for player in range(players):
#                     if curr_player_results[0,player] != -1:
#                         curr_player_results[0,player] = 1
#             else:
#                 for player in range(players):
#                     if total_up(player_hands[player]) > total_up(dealer_hand):
#                         if total_up(player_hands[player]) <= 21:
#                             print("win")
#                             curr_player_results[0,player] = 1
#                     elif total_up(player_hands[player]) == total_up(dealer_hand):
#                         print("tie")
#                         curr_player_results[0,player] = 0
#                     else:
#                         print("lose")
#                         curr_player_results[0,player] = -1
