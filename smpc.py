import random

class Party:
    def __init__(self, name, secret, modulus=1000):
        self.name = name
        self.secret = secret
        self.modulus = modulus
        self.share1 = None
        self.share2 = None

    def generate_shares(self):
        """Generates two shares of the secret using additive secret sharing."""
        self.share1 = random.randint(0, self.modulus - 1)
        self.share2 = (self.secret - self.share1) % self.modulus
        return self.share1, self.share2

def secure_addition(party1, party2):
    """Performs secure addition of two parties' secrets using SMPC."""
    # Each party generates their shares
    s1_p1, s2_p1 = party1.generate_shares()
    s1_p2, s2_p2 = party2.generate_shares()

    print(f"{party1.name}'s shares: Share1 = {s1_p1}, Share2 = {s2_p1}")
    print(f"{party2.name}'s shares: Share1 = {s1_p2}, Share2 = {s2_p2}")

    # Share exchange
    # party1 keeps s1_p1 and receives s2_p2 from party2
    # party2 keeps s1_p2 and receives s2_p1 from party1
    total_share_p1 = (s1_p1 + s2_p2) % party1.modulus
    total_share_p2 = (s1_p2 + s2_p1) % party2.modulus

    print(f"{party1.name}'s total share: {total_share_p1}")
    print(f"{party2.name}'s total share: {total_share_p2}")

    # Compute the final sum
    total = (total_share_p1 + total_share_p2) % party1.modulus
    return total

# Define secrets
alice_secret = 123
bob_secret = 456

alice = Party("Alice", alice_secret)
bob = Party("Bob", bob_secret)

# Perform SMPC to compute the sum
computed_sum = secure_addition(alice, bob)
expected_sum = (alice_secret + bob_secret) % alice.modulus

print(f"\nComputed Sum: {computed_sum}")
print(f"Expected Sum: {expected_sum}")

if computed_sum == expected_sum:
    print("SMPC computation was executed correctly.")
else:
    print("SMPC computation failed.")
