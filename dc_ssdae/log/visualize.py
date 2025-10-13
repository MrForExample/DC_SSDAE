import matplotlib.pyplot as plt


def show_generation_result(x_result, img_conditions=None, title="Generated samples"):
    n_samples = len(x_result)
    fig, axes = plt.subplots(n_samples, 2, figsize=(3 * 2, 3 * n_samples), sharex=True, sharey=True)

    if img_conditions is not None:
        for i_sample in range(n_samples):
            ax = axes[i_sample, 0]
            ax.imshow(img_conditions[i_sample].cpu().permute(1, 2, 0))
            ax.axis("off")
            ax.set_title("Image condition")

    for i_sample in range(n_samples):
        ax = axes[i_sample, 1]
        ax.imshow(x_result[i_sample].cpu().permute(1, 2, 0))
        ax.axis("off")
        ax.set_title(f"Reconstructed i={i_sample}")

    fig.tight_layout()
    if title:
        fig.suptitle(title)
    return fig
