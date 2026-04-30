"""
Full Lifecycle Speaker Recognition Animation (Fixed for np.dot formatting)

Shows:
1) Enrollment (samples appear one by one)
2) Centroid formation (average embedding)
3) Unit-sphere cosine similarity
4) Identification with a moving test embedding
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# -----------------------------
# Utilities
# -----------------------------

def normalize(v):
    return v / np.linalg.norm(v)


def make_samples(center, n=20, noise=0.15):
    samples = []
    for _ in range(n):
        v = center + noise * np.random.randn(2)
        samples.append(normalize(v))
    return np.array(samples)


# -----------------------------
# Speaker identities
# -----------------------------

np.random.seed(42)

alice_true = normalize(np.array([0.7, 0.7]))
bob_true   = normalize(np.array([-0.8, 0.6]))
eve_true   = normalize(np.array([-0.2, -0.9]))

alice_samples = make_samples(alice_true)
bob_samples   = make_samples(bob_true)
eve_samples   = make_samples(eve_true)

# -----------------------------
# Test embedding (identify phase)
# -----------------------------

angles = np.linspace(-120, 45, 80) * np.pi / 180
test_embeddings = np.array([normalize([np.cos(a), np.sin(a)]) for a in angles])

# -----------------------------
# Plot setup
# -----------------------------

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_title(
    "Speaker Recognition: Enrollment → Centroids → Identification",
    fontsize=12
)

# Unit circle
theta = np.linspace(0, 2*np.pi, 400)
ax.plot(np.cos(theta), np.sin(theta), linewidth=1)

# Scatter placeholders
alice_scatter = ax.scatter([], [], alpha=0.7)
bob_scatter   = ax.scatter([], [], alpha=0.7)
eve_scatter   = ax.scatter([], [], alpha=0.7)

# Centroid markers
alice_star, = ax.plot([], [], "*", markersize=14)
bob_star,   = ax.plot([], [], "*", markersize=14)
eve_star,   = ax.plot([], [], "*", markersize=14)

alice_label = ax.text(0, 0, "", weight="bold")
bob_label   = ax.text(0, 0, "", weight="bold")
eve_label   = ax.text(0, 0, "", weight="bold")

# Test vector
test_vector, = ax.plot([], [], linewidth=3)

# Info text
info_text = ax.text(-0.30, 0, "", fontsize=10)


# -----------------------------
# Animation parameters
# -----------------------------

ENROLL_FRAMES_PER_SPEAKER = 20
CENTROID_FRAME = 65
TEST_START_FRAME = 70

TOTAL_FRAMES = 3 * ENROLL_FRAMES_PER_SPEAKER + 10 + len(test_embeddings)


# -----------------------------
# Animation logic
# -----------------------------

def init():
    return (
        alice_scatter, bob_scatter, eve_scatter,
        alice_star, bob_star, eve_star,
        test_vector, info_text
    )


def update(frame):
    # -------------------------
    # Enrollment phase
    # -------------------------
    if frame < ENROLL_FRAMES_PER_SPEAKER:
        alice_scatter.set_offsets(alice_samples[:frame])
        info_text.set_text("Enrolling Alice samples")

    elif frame < 2 * ENROLL_FRAMES_PER_SPEAKER:
        bob_frame = frame - ENROLL_FRAMES_PER_SPEAKER
        bob_scatter.set_offsets(bob_samples[:bob_frame])
        info_text.set_text("Enrolling Bob samples")

    elif frame < 3 * ENROLL_FRAMES_PER_SPEAKER:
        eve_frame = frame - 2 * ENROLL_FRAMES_PER_SPEAKER
        eve_scatter.set_offsets(eve_samples[:eve_frame])
        info_text.set_text("Enrolling Eve samples")

    # -------------------------
    # Centroid formation
    # -------------------------
    elif frame < TEST_START_FRAME:


        waiting_frame = frame - 10 * ENROLL_FRAMES_PER_SPEAKER
        if waiting_frame < (CENTROID_FRAME - 10 * ENROLL_FRAMES_PER_SPEAKER):
            info_text.set_text("Computing centroids for all speakers...")
            return (
                alice_scatter, bob_scatter, eve_scatter,
                alice_star, bob_star, eve_star,
                test_vector, info_text
            )

        # Compute centroids as mean of enrolled samples
        alice_centroid = normalize(alice_scatter.get_offsets().mean(axis=0))
        bob_centroid   = normalize(bob_scatter.get_offsets().mean(axis=0))
        eve_centroid   = normalize(eve_scatter.get_offsets().mean(axis=0))

        # Update stars (use sequences)
        alice_label.set_text("Alice")
        alice_label.set_position(alice_centroid + 0.05)
        alice_star.set_data([alice_centroid[0]], [alice_centroid[1]])

        
        bob_label.set_text("Bob")
        bob_label.set_position(bob_centroid + 0.05)
        bob_star.set_data([bob_centroid[0]], [bob_centroid[1]])

        eve_label.set_text("Eve")
        eve_label.set_position(eve_centroid + 0.05)
        eve_star.set_data([eve_centroid[0]], [eve_centroid[1]])
        
        info_text.set_text("Averaging embeddings → centroids")

    # -------------------------
    # Identification phase
    # -------------------------
    else:
        idx = frame - TEST_START_FRAME
        if idx >= len(test_embeddings):
            idx = len(test_embeddings) - 1

        test_vec = test_embeddings[idx]
        test_vector.set_data([0, test_vec[0]], [0, test_vec[1]])

        # Compute cosine similarity and convert to float
        score_a = float(np.dot(test_vec, normalize(alice_star.get_data())))
        score_b = float(np.dot(test_vec, normalize(bob_star.get_data())))
        score_e = float(np.dot(test_vec, normalize(eve_star.get_data())))

        best = max([("Alice", score_a), ("Bob", score_b), ("Eve", score_e)], key=lambda x: x[1])

        info_text.set_text(
            f"Identifying speaker\n"
            f"cos(test, Alice) = {score_a:.2f}\n"
            f"cos(test, Bob)   = {score_b:.2f}\n"
            f"cos(test, Eve)   = {score_e:.2f}\n\n"
            f"→ Identified as: {best[0]}"
        )

    return (
        alice_scatter, bob_scatter, eve_scatter,
        alice_star, bob_star, eve_star,
        test_vector, info_text
    )


ani = FuncAnimation(
    fig,
    update,
    frames=TOTAL_FRAMES,
    init_func=init,
    blit=True
)

# # -----------------------------
# # Save animation
# # -----------------------------

# output_file = "./speaker_embedding_full_lifecycle_fixed.gif"
# ani.save(output_file, writer="pillow", fps=6)

# print(f"Animation saved as: {output_file}")


# -----------------------------
# Save animation as video
# -----------------------------
from matplotlib.animation import FFMpegWriter

output_file = "speaker_embedding_full_lifecycle_fixed.mp4"
writer = FFMpegWriter(fps=6, metadata=dict(artist='Vinay'), bitrate=1800)

ani.save(output_file, writer=writer)
print(f"Animation saved as video: {output_file}")