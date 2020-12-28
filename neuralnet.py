import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout
if __name__ == "__main__":
    # This function lists out all permutations of ace values in the array sum_array
    # For example, if you have 2 aces, there are 4 permutations:
    #     [[1,1], [1,11], [11,1], [11,11]]
    # These permutations lead to 3 unique sums: [2, 12, 22]
    # Of these 3, only 2 are <=21 so they are returned: [2, 12]
    def get_ace_values(temp_list):
        sum_array = np.zeros((2**len(temp_list), len(temp_list)))
        # This loop gets the permutations
        for i in range(len(temp_list)):
            n = len(temp_list) - i
            half_len = int(2**n * 0.5)
            for rep in range(int(sum_array.shape[0]/half_len/2)):
                sum_array[rep*2**n : rep*2**n+half_len, i] = 1
                sum_array[rep*2**n+half_len : rep*2**n+half_len*2, i] = 11
        # Only return values that are valid (<=21)
        # return list(set([int(s) for s in np.sum(sum_array, axis=1) if s<=21]))
        return [int(s) for s in np.sum(sum_array, axis=1)]

    # Convert num_aces, an int to a list of lists
    # For example if num_aces=2, the output should be [[1,11],[1,11]]
    # I require this format for the get_ace_values function
    def ace_values(num_aces):
        temp_list = []
        for i in range(num_aces):
            temp_list.append([1,11])
        return get_ace_values(temp_list)

    # Make a deck
    def make_decks(num_decks, card_types):
        new_deck = []
        for i in range(num_decks):
            for j in range(4):
                new_deck.extend(card_types)
        random.shuffle(new_deck)
        return new_deck

    # Total up value of hand
    def total_up(hand):
        aces = 0
        total = 0

        for card in hand:
            if card != 'A':
                total += card
            else:
                aces += 1

        # Call function ace_values to produce list of possible values for aces in hand
        ace_value_list = ace_values(aces)
        final_totals = [i+total for i in ace_value_list if i+total<=21]

        if final_totals == []:
            return min(ace_value_list) + total
        else:
            return max(final_totals)

    # Play a game of blackjack (after the cards are dealt)
    def play_game(dealer_hand, player_hands, blackjack, curr_player_results, dealer_cards, hit_stay, card_count, dealer_bust):
        action = 0
        # Dealer checks for 21
        if set(dealer_hand) == blackjack:
            for player in range(players):
                if set(player_hands[player]) != blackjack:
                    curr_player_results[0,player] = -1
                else:
                    curr_player_results[0,player] = 0
        else:
            for player in range(players):
                # Players check for 21
                if set(player_hands[player]) == blackjack:
                    curr_player_results[0,player] = 1
                else:
                    # Hit randomly, check for busts
                    if (hit_stay >= 0.5) and (total_up(player_hands[player]) != 21):
                        player_hands[player].append(dealer_cards.pop(0))
                        card_count[player_hands[player][-1]] += 1

                        action = 1
                        live_total.append(total_up(player_hands[player]))
                        if total_up(player_hands[player]) > 21:
                            curr_player_results[0,player] = -1

        # Dealer hits based on the rules
        card_count[dealer_hand[-1]] += 1
        while total_up(dealer_hand) < 17:
            dealer_hand.append(dealer_cards.pop(0))
            card_count[dealer_hand[-1]] += 1

        # Compare dealer hand to players hand but first check if dealer busted
        if total_up(dealer_hand) > 21:
            dealer_bust.append(1)
            for player in range(players):
                if curr_player_results[0,player] != -1:
                    curr_player_results[0,player] = 1
        else:
            dealer_bust.append(0)
            for player in range(players):
                if total_up(player_hands[player]) > total_up(dealer_hand):
                    if total_up(player_hands[player]) <= 21:
                        curr_player_results[0,player] = 1
                elif total_up(player_hands[player]) == total_up(dealer_hand):
                    curr_player_results[0,player] = 0
                else:
                    curr_player_results[0,player] = -1

        return curr_player_results, dealer_cards, action, card_count, dealer_bust


    stacks = 40000
    players = 1
    num_decks = 1

    card_types = ['A',2,3,4,5,6,7,8,9,10,10,10,10]

    dealer_card_feature = []
    player_card_feature = []
    player_live_total = []
    player_live_action = []
    player_results = []
    dealer_bust = []

    first_game = True
    prev_stack = 0
    stack_num_list = []
    new_stack = []
    card_count_list = []
    games_played_with_stack = []

    for stack in range(stacks):
        games_played = 0

        # Make a dict for keeping track of the count for a stack
        card_count = {2: 0,
                      3: 0,
                      4: 0,
                      5: 0,
                      6: 0,
                      7: 0,
                      8: 0,
                      9: 0,
                      10: 0,
                      'A': 0}

        blackjack = set(['A',10])
        dealer_cards = make_decks(num_decks, card_types)
        while len(dealer_cards) > 20:

            curr_player_results = np.zeros((1,players))

            dealer_hand = []
            player_hands = [[] for player in range(players)]
            live_total = []
            live_action = []

            # Deal FIRST card
            for player, hand in enumerate(player_hands):
                player_hands[player].append(dealer_cards.pop(0))
                card_count[player_hands[player][-1]] += 1

            dealer_hand.append(dealer_cards.pop(0))
            card_count[dealer_hand[-1]] += 1

            # Deal SECOND card
            for player, hand in enumerate(player_hands):
                player_hands[player].append(dealer_cards.pop(0))
                card_count[player_hands[player][-1]] += 1

            dealer_hand.append(dealer_cards.pop(0))

            # Record the player's live total after cards are dealt
            live_total.append(total_up(player_hands[player]))

            if stack < stacks/2:
                hit_stay = 1
            else:
                hit_stay = 0

            curr_player_results, dealer_cards, action, card_count, dealer_bust = play_game(dealer_hand, player_hands,
                                                                                           blackjack, curr_player_results,
                                                                                           dealer_cards, hit_stay, card_count,
                                                                                           dealer_bust)

            # Track features
            dealer_card_feature.append(dealer_hand[0])
            player_card_feature.append(player_hands)
            player_results.append(list(curr_player_results[0]))
            player_live_total.append(live_total)
            player_live_action.append(action)

            # Update card count list with most recent game's card count
            if stack != prev_stack:
                new_stack.append(1)
            else:
                new_stack.append(0)
                if first_game == True:
                    first_game = False
                else:
                    games_played += 1

            stack_num_list.append(stack)
            games_played_with_stack.append(games_played)
            card_count_list.append(card_count.copy())
            prev_stack = stack

    model_df = pd.DataFrame()
    model_df['dealer_card'] = dealer_card_feature
    model_df['player_total_initial'] = [total_up(i[0][0:2]) for i in player_card_feature]
    model_df['hit?'] = player_live_action

    has_ace = []
    for i in player_card_feature:
        if ('A' in i[0][0:2]):
            has_ace.append(1)
        else:
            has_ace.append(0)
    model_df['has_ace'] = has_ace

    dealer_card_num = []
    for i in model_df['dealer_card']:
        if i=='A':
            dealer_card_num.append(11)
        else:
            dealer_card_num.append(i)
    model_df['dealer_card_num'] = dealer_card_num

    model_df['Y'] = [i[0] for i in player_results]
    lose = []
    for i in model_df['Y']:
        if i == -1:
            lose.append(1)
        else:
            lose.append(0)
    model_df['lose'] = lose

    correct = []
    for i, val in enumerate(model_df['lose']):
        if val == 1:
            if player_live_action[i] == 1:
                correct.append(0)
            else:
                correct.append(1)
        else:
            if player_live_action[i] == 1:
                correct.append(1)
            else:
                correct.append(0)
    model_df['correct_action'] = correct

    # Make a new version of model_df that has card counts
    card_count_df = pd.concat([pd.DataFrame(new_stack, columns=['new_stack']),
                               pd.DataFrame(games_played_with_stack, columns=['games_played_with_stack']),
                               pd.DataFrame.from_dict(card_count_list),
                               pd.DataFrame(dealer_bust, columns=['dealer_bust'])], axis=1)
    model_df = pd.concat([model_df, card_count_df], axis=1)

    data = 1 - (model_df.groupby(by='dealer_card').sum()['lose'] / \
                model_df.groupby(by='dealer_card').count()['lose'])

    fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.barplot(x=data.index,
                     y=data.values)
    ax.set_xlabel("Dealer's Card",fontsize=16)
    ax.set_ylabel("Probability of Tie or Win",fontsize=16)

    plt.tight_layout()
    plt.savefig(fname='dealer_card_probs', dpi=150)



    data = 1 - (model_df.groupby(by='player_total_initial').sum()['lose'] / \
                model_df.groupby(by='player_total_initial').count()['lose'])

    fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.barplot(x=data[:-1].index,
                     y=data[:-1].values)
    ax.set_xlabel("Player's Hand Value",fontsize=16)
    ax.set_ylabel("Probability of Tie or Win",fontsize=16)

    plt.tight_layout()
    plt.savefig(fname='player_hand_probs', dpi=150)

    model_df.groupby(by='has_ace').sum()['lose'] / model_df.groupby(by='has_ace').count()['lose']


    pivot_data = model_df[model_df['player_total_initial'] != 21]

    losses_pivot = pd.pivot_table(pivot_data, values='lose',
                                  index=['dealer_card_num'],
                                  columns = ['player_total_initial'],
                                  aggfunc = np.sum)

    games_pivot =  pd.pivot_table(pivot_data, values='lose',
                                  index=['dealer_card_num'],
                                  columns = ['player_total_initial'],
                                  aggfunc = 'count')

    heat_data = 1 - losses_pivot.sort_index(ascending=False) / games_pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(16,8))
    sns.heatmap(heat_data, square=False, cmap="PiYG");

    ax.set_xlabel("Player's Hand Value",fontsize=16)
    ax.set_ylabel("Dealer's Card",fontsize=16)

    plt.savefig(fname='heat_map_random', dpi=150)


    # Train a neural net to play blackjack

    # Set up variables for neural net
    feature_list = [i for i in model_df.columns if i not in ['dealer_card',
                                                             'Y','lose',
                                                             'correct_action',
                                                             'dealer_bust',
                                                             'dealer_bust_pred',
                                                             'new_stack', 'games_played_with_stack',
                                                             2,3,4,5,6,7,8,9,10,'A',
                                                             'blackjack?'
                                                             ]]
    train_X = np.array(model_df[feature_list])
    train_Y = np.array(model_df['correct_action']).reshape(-1,1)

    print(feature_list)
    print('\n')

    # Set up a neural net with 5 layers
    model = Sequential()
    # model.add(Dense(train_X.shape[1]+1))
    model.add(Dense(16))
    model.add(Dense(128))
    model.add(Dense(32))
    model.add(Dense(8))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.fit(train_X, train_Y, epochs=200, batch_size=256, verbose=1)

    pred_Y_train = model.predict(train_X)
    actuals = train_Y[:,-1]


