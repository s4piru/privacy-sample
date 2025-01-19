import random


# For demonstration, use the prime p = 2^127 - 1
# (In practice, use larger primes and secure generation methods)
p = 2**127 - 1

# Choose base elements g and h within the field defined by p
# (Ensure g and h are generators with unknown discrete logarithm relationship)
g = 3
h = 5

def commit(value: int, nonce: int) -> int:
    """
    Computes the commitment c = (g^value * h^nonce) mod p using the value and random nonce.
    
    Parameters:
    - value (int): The value to commit to (1 <= value <= 50).
    - nonce (int): A random nonce (1 <= nonce < p).
    
    Returns:
    - int: The resulting commitment.
    """
    return (pow(g, value, p) * pow(h, nonce, p)) % p


def prove_secret_in_range(x: int):
    """
    Simulates a Zero-Knowledge Proof where Alice proves to Bob that her secret integer x 
    is within the range 1 to 50 without revealing x.
    
    Parameters:
    - x (int): Alice's secret integer (1 <= x <= 50).
    """
    
    if not (1 <= x <= 50):
        raise ValueError("Secret x must be within the range [1, 50].")
    

    # Step A: Alice creates commitments for all integers [1..50]
    commits = []
    secrets = []

    for value in range(1, 51):
        nonce = random.randint(1, p-1)
        c = commit(value, nonce)
        commits.append(c)
        secrets.append((value, nonce))


    # Step B: Shuffle the commitments
    index_list = list(range(50))
    random.shuffle(index_list)
    
    shuffled_commits = [commits[idx] for idx in index_list]
    shuffled_secrets = [secrets[idx] for idx in index_list]

    # Determine the position of x in the shuffled list
    original_index_of_x = x - 1  # x=1 maps to index 0, etc.
    shuffled_position_of_x = index_list.index(original_index_of_x)

    # Step C: Alice presents the shuffled commitments to Bob
    # Alice sends all shuffled_commits to Bob without revealing the order or which corresponds to which i
    
    # Step D: Alice discloses all but the commitment for x
    open_indices = [i for i in range(50) if i != shuffled_position_of_x]
    
    # Prepare the information to be disclosed to Bob
    opened_info = []
    for idx in open_indices:
        c = shuffled_commits[idx]
        value, nonce = shuffled_secrets[idx]
        opened_info.append((c, value, nonce))


    # Step E: Bob verifies the disclosed commitments
    for (c_opened, value, nonce) in opened_info:
        c_recomputed = commit(value, nonce)
        if c_opened != c_recomputed:
            print("[Bob] Commitment verification failed!")
            return
    
    # At this point, Bob has verified 49 commitments corresponding to integers other than x.
    # He knows the remaining commitment corresponds to x being within [1..50], but does not know x.

    print(f"[Alice] Secret x is within the range [1, 50]. (Actual x: {x})")
    print(f"[Alice] The commitment corresponding to x is at shuffled position {shuffled_position_of_x} (not disclosed)")
    print(f"[Bob] Received all 50 commitments. Verified 49 commitments are correct and correspond to known values.")
    print("[Bob] The remaining 1 commitment corresponds to the secret x, but its actual value is unknown.")
    print("[Conclusion] Bob is convinced that x is within the range 1 to 50 without knowing the exact value of x (Zero-Knowledge)!")


if __name__ == "__main__":
    # Alice selects a secret x randomly between 1 and 50
    secret_x = random.randint(1, 50)
    
    # Execute the Zero-Knowledge Proof demonstration
    prove_secret_in_range(secret_x)
