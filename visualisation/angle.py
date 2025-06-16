import random
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

def plot_2d_vectors(vectors, max_radius=100):
    plt.figure(figsize=(6, 6))
    plt.scatter(vectors[:, 0], vectors[:, 1], label='2D Vectors')
    plt.xlim(0, max_radius)
    plt.ylim(0, max_radius)
    plt.title('2D Quarter Circle Vectors')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

def plot_full_cone_with_vectors(vectors, max_radius=100):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Draw the full cone surface
    r_vals = np.linspace(0, max_radius, 50)
    theta_vals = np.linspace(0, 2 * np.pi, 60)
    R, T = np.meshgrid(r_vals, theta_vals)
    Xc = R * np.cos(T)
    Yc = R * np.sin(T)
    Zc = R  # For a 45° cone: z = r

    ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color='gray', edgecolor='none')

    # 2. Map the 2D quarter-circle vectors onto the cone wall
    r = np.linalg.norm(vectors, axis=1)
    theta = np.linspace(0, 2 * np.pi, len(vectors) + 1)[:len(vectors)]
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    Z = r

    # 3. Plot the vectors on the cone wall
    ax.scatter(X, Y, Z, color='red', label='Mapped vectors')
    
    # 4. Draw arrows (quiver) from origin to vector tips
    # ax.quiver(np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z),
    #           X, Y, Z, length=1, normalize=False, color='blue', arrow_length_ratio=0.05)

    # 5. Mark the origin
    ax.scatter([0], [0], [0], color='black', s=50, label='Origin (0,0,0)')

    # ✅ 6. Draw vertical line through center of cone (Z-axis)
    ax.plot([0, 0], [0, 0], [0, max_radius], color='green', linewidth=2, label='Cone axis')

    ax.set_title('Full 3D Cone with Mapped Vectors')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(0, max_radius)
    ax.legend()
    plt.tight_layout()
    plt.show()

def get_vocabulary_info(model, step, n=10, agent_filter=None, agent_comparison_filter=None, ax=None):      
    if model.datacollector_step_size != 1:
        step = step // model.datacollector_step_size

    df = model.datacollector.get_model_vars_dataframe()
    full_vocabulary = df["full_vocabulary"].iloc[step]
    exemplar_indices = df["full_indices"].iloc[step]
    ownership_indices = df["full_vocabulary_owernship"].iloc[step]
    labels = model.tokens[:n]

    # Get only those indices which correspond to the top n
    eligible_indices = [ exemplar_index for exemplar_index in range(full_vocabulary.shape[0]) if exemplar_indices[exemplar_index] < n and (agent_filter is None or ownership_indices[exemplar_index] == agent_filter) ]

    if agent_comparison_filter is not None:
        eligible_indices_comparison = [ exemplar_index for exemplar_index in range(full_vocabulary.shape[0]) if exemplar_indices[exemplar_index] < n and ownership_indices[exemplar_index] == agent_comparison_filter ]

    vocabulary = full_vocabulary[eligible_indices, :]
    indices = exemplar_indices[eligible_indices]
    border = len(indices)

    indices_comparison = None
    if agent_comparison_filter is not None:
        vocabulary_comparison = full_vocabulary[eligible_indices_comparison, :]
        indices_comparison = exemplar_indices[eligible_indices_comparison]

        vocabulary = np.vstack([ vocabulary, vocabulary_comparison ])
        # indices = np.concatenate([ indices, indices_comparison ])

    random_index = random.choice(range(vocabulary.shape[0]))
    random_vector = vocabulary[random_index,:]

    # Get colour for each data point
    colours = [ "red", "green", "blue", "brown", "yellow", "purple", "black", "pink", "grey", "teal" ]

    return vocabulary, random_vector, indices, colours, labels, indices_comparison, border

def make_angle_vocabulary_plot_2d(model, step, n=10, agent_filter=None, agent_comparison_filter=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))

    max_radius = model.value_ceil
    vocabulary, random_vector, indices, colours, labels, indices_comparison, border = get_vocabulary_info(model, step, n, agent_filter, agent_comparison_filter)    
    x = vocabulary[:,0]
    y = vocabulary[:,1]
    
    colours = ListedColormap(colours)
    
    scatter = ax.scatter(x[:border], y[:border], c=indices, cmap=colours, marker="^", alpha=0.9)
    if agent_comparison_filter is not None:
        scatter = ax.scatter(x[border:], y[border:], c=indices_comparison, cmap=colours, marker="v", alpha=0.5)

    neighbourhood = plt.Circle(random_vector, model.neighbourhood_size, color="grey", alpha=0.2)
    ax.add_patch(neighbourhood)
    
    ax.set_xlim(0, max_radius)
    ax.set_ylim(0, max_radius)

    ax.legend(handles=scatter.legend_elements()[0], labels=labels)

    ax.set_title(f"Plot of exemplars (t = {step})")

    return ax

def make_angle_vocabulary_plot_3d(model, step, n=10, agent_filter=None, agent_comparison_filter=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')

    max_radius = model.value_ceil
    vocabulary, random_vector, indices, colours, labels, indices_comparison, border = get_vocabulary_info(model, step, n, agent_filter, agent_comparison_filter)

    # 1. Draw the full cone surface
    theta_vals = np.linspace(0, 2 * np.pi, max_radius)
    r_vals = np.linspace(0, max_radius, 50)
    R, T = np.meshgrid(r_vals, theta_vals)
    Xc = R * np.cos(T)
    Yc = R * np.sin(T)
    Zc = max_radius - R  # For a 45° cone: z = r

    ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='gray', edgecolor='none')

    # 2. Map the 2D quarter-circle vectors onto the cone wall
    # OLD
    # r = np.linalg.norm(vocabulary, axis=1)
    # theta = np.linspace(0, 2 * np.pi, len(vocabulary) + 1)[:len(vocabulary)]
    # X = r * np.cos(theta)
    # Y = r * np.sin(theta)
    # Z = r

    # NEW
    x = vocabulary[:,0]
    y = vocabulary[:,1]
    r = np.linalg.norm(vocabulary, axis=1)
    theta = np.arctan2(y, x) # angle in [0, π/2]
    theta_cone = (theta / (0.5 * np.pi)) * 2 * np.pi # map to [0, 2π]

    X = r * np.cos(theta_cone)
    Y = r * np.sin(theta_cone)
    Z = np.int64(max_radius) - r
    
    colours = ListedColormap(colours)

    # 3. Plot the vectors on the cone wall
    ax.scatter(X[:border], Y[:border], Z[:border], label='Mapped vectors', c=indices, cmap=colours, marker="^", alpha=0.5)
    if agent_comparison_filter is not None:
        scatter = ax.scatter(X[border:], Y[border:], Z[border:], c=indices_comparison, cmap=colours, marker="v", alpha=0.5)
    
    # 4. Draw arrows (quiver) from origin to vector tips
    # ax.quiver(np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z),
    #           X, Y, Z, length=1, normalize=False, color='blue', arrow_length_ratio=0.05)

    # 5. Mark the origin
    ax.scatter([0], [0], [max_radius], color='black', s=50, label='Origin (0,0,0)')

    # ✅ 6. Draw vertical line through center of cone (Z-axis)
    ax.plot([0, 0], [0, 0], [0, max_radius], color='green', linewidth=2, label='Cone axis')

    ax.set_title('Full 3D Cone with Mapped Vectors')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(0, max_radius)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return ax

def make_angle_vocabulary_plot_3d_interactive(model, step, n=10, agent_filter=None, agent_comparison_filter=None):
    max_radius = model.value_ceil
    vocabulary, random_vector, indices, colours, labels, indices_comparison, border = get_vocabulary_info(model, step, n, agent_filter, agent_comparison_filter)

    # Cone surface
    theta_vals = np.linspace(0, 2 * np.pi, 100)
    r_vals = np.linspace(0, max_radius, 50)
    R, T = np.meshgrid(r_vals, theta_vals)
    Xc = R * np.cos(T)
    Yc = R * np.sin(T)
    Zc = max_radius - R  # 45° cone

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=Xc, y=Yc, z=Zc,
        opacity=0.5,
        showscale=False,
        colorscale=[[0, 'lightgray'], [1, 'lightgray']],
        name='Cone'
    ))

    # Map 2D points onto cone
    x = vocabulary[:,0]
    y = vocabulary[:,1]
    r = np.linalg.norm(vocabulary, axis=1)
    theta = np.arctan2(y, x)
    theta_cone = (theta / (0.5 * np.pi)) * 2 * np.pi

    X = r * np.cos(theta_cone)
    Y = r * np.sin(theta_cone)
    Z = max_radius - r

    # Vectors
    # fig.add_trace(go.Scatter3d(
    #     x=X[:border], y=Y[:border], z=Z[:border],
    #     mode='markers',
    #     marker=dict(size=4, color=indices, colorscale=colours),
    #     name='Mapped vectors'
    # ))

    
    fig.add_trace(go.Scatter3d(
        x=X[:border], y=Y[:border], z=Z[:border],
        mode='text',
        text=[str(i) for i in indices[:border]],
        textposition='middle center',
        textfont=dict(size=10, color='blue'),
        name='Mapped indices'
    ))

    if agent_comparison_filter is not None:
        fig.add_trace(go.Scatter3d(
            x=X[border:], y=Y[border:], z=Z[border:],
            mode='markers',
            marker=dict(size=4, color=indices_comparison, colorscale=colours),
            name='Comparison vectors'
        ))

    # Origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[max_radius],
        mode='markers',
        marker=dict(color='black', size=6),
        name='Origin'
    ))

    # Vertical axis
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, max_radius],
        mode='lines',
        line=dict(color='green', width=4),
        name='Cone axis'
    ))

    fig.update_layout(
        title='3D Cone with Interactive Mapping',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=True
    )

    fig.show()
