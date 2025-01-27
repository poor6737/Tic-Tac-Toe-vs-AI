import random

# Initialize the Q-table
Q = {}

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate (probability of choosing a random action)

# Function to check for a winner
def check_winner(board):
    winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
                            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
                            (0, 4, 8), (2, 4, 6)]  # diagonals

    for a, b, c in winning_combinations:
        if board[a] == board[b] == board[c] != 0:
            return board[a]  # Return the winner (1 or -1)
    return 0  # No winner yet

# Function to check for a draw
def is_draw(board):
    return all(cell != 0 for cell in board)

# Function to update the Q-table
def update_q_table(state, action, reward, next_state):
    state = tuple(state)  # Convert state to tuple to make it hashable
    next_state = tuple(next_state)  # Convert next_state to tuple

    max_future_q = max(Q.get(next_state, [0] * 9))  # Best next action’s Q-value
    current_q = Q.get(state, [0] * 9)[action]  # Current action’s Q-value

    # Update the Q-value
    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
    if state not in Q:
        Q[state] = [0] * 9  # Initialize Q-values if not already initialized
    Q[state][action] = new_q

# Function to choose an action (move)
def choose_action(board):
    available_actions = [i for i, x in enumerate(board) if x == 0]  # Only available positions

    if random.uniform(0, 1) < epsilon:
        # Explore: Randomly choose from available actions
        return random.choice(available_actions)
    else:
        # Exploit: Choose the action with the highest Q-value
        state = tuple(board)
        return max(available_actions, key=lambda x: Q.get(state, [0] * 9)[x])  # Only choose from available actions

# Function to print the current state of the board
def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    print(f"\n{symbols[board[0]]} | {symbols[board[1]]} | {symbols[board[2]]}")
    print("--+---+--")
    print(f"{symbols[board[3]]} | {symbols[board[4]]} | {symbols[board[5]]}")
    print("--+---+--")
    print(f"{symbols[board[6]]} | {symbols[board[7]]} | {symbols[board[8]]}\n")

# Function for human vs AI gameplay
def play_against_ai():
    board = [0] * 9
    current_player = 1  # AI starts
    while True:
        print_board(board)

        if current_player == 1:
            print("AI's turn:")
            action = choose_action(board)  # AI makes move
            print(f"AI chooses position {action}")
        else:
            # Human player makes move
            valid_move = False
            while not valid_move:
                try:
                    action = input("Your turn! Enter a position (0-8): ")
                    action = int(action)  # Try to convert input to integer
                    if action < 0 or action > 8:
                        print("Invalid position! Choose a position between 0 and 8.")
                    elif board[action] != 0:
                        print("That position is already taken. Try again.")
                    else:
                        valid_move = True  # The move is valid
                except ValueError:
                    print("Invalid input! Please enter a number between 0 and 8.")  # If input is not a valid integer

        # Update the board with the chosen action
        board[action] = current_player

        # Check for a winner
        winner = check_winner(board)
        if winner != 0:
            print_board(board)
            reward = 1 if winner == current_player else -1
            update_q_table(tuple(board), action, reward, board)
            print(f"Player {winner} wins!")
            break

        if is_draw(board):
            print_board(board)
            update_q_table(tuple(board), action, 0, board)
            print("It's a draw!")
            break

        # Update Q-values for the AI after it makes a move
        update_q_table(tuple(board), action, 0, board)

        # Switch player
        current_player = -current_player  # Switch player

# Function to train the AI
def train_ai(num_games=1000):
    print(f"Training AI with {num_games} games...")
    for _ in range(num_games):
        board = [0] * 9
        current_player = 1  # AI starts
        while True:
            action = choose_action(board)  # AI chooses an action
            board[action] = current_player  # AI makes its move

            # Check for a winner
            winner = check_winner(board)
            if winner != 0:
                reward = 1 if winner == 1 else -1
                update_q_table(tuple(board), action, reward, board)
                break

            if is_draw(board):
                update_q_table(tuple(board), action, 0, board)
                break

            # Update Q-values after AI's move
            update_q_table(tuple(board), action, 0, board)

            # Switch player
            current_player = -current_player  # Switch player

# Start by training the AI
train_ai(num_games=10000)  # Train AI with 10,000 games

# Now, start the AI vs Human game
print("\nAI vs Human Player:")
play_against_ai()  # Play the game with human vs AI
