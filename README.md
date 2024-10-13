# Movie Recommender System

## Overview

Welcome to the Movie Recommender System project! This system leverages the Movies dataset from Kaggle to deliver personalized movie recommendations. By utilizing a blend of collaborative filtering, content-based filtering, and hybrid methods, we aim to enhance the movie-watching experience. Additionally, we include a simple feature that highlights the top 10 movies in each genre. To take things up a notch, we implemented Generative Adversarial Networks (GANs) to generate extra user ratings, enriching our dataset for improved model performance.

## Key Features

- **Collaborative Filtering**: Our system identifies user preferences based on the ratings of similar users, helping you discover movies that align with your tastes.
- **Content-Based Filtering**: By analyzing movie attributes like genre, director, and cast, this method provides recommendations tailored to your interests.

- **Hybrid Recommendations**: Combining the strengths of collaborative and content-based filtering, this approach offers a more comprehensive recommendation experience.

- **Simple Recommender**: Need quick suggestions? Our simple recommender lists the top 10 movies for each genre, making it easy to find popular picks.

- **GANs for Data Augmentation**: To enrich our dataset, we applied GANs to generate additional user ratings, allowing us to train the model on a more diverse set of data.

## Dataset

We use the [Movies dataset from Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/code), which contains vital information such as:

- Movie titles
- Genres
- User ratings
- Details about the cast and crew

## Getting Started

To get the project up and running, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/movie-recommender.git
   cd movie-recommender
   ```

## How to Use

1. Load and preprocess the dataset.
2. Train your recommender model, which utilizes **hybrid recommendations**—combining both content-based and collaborative filtering methods for better accuracy.
3. Additionally, a **simple model** is available that provides the top movies for each genre, making it easy to explore popular options.
4. Generate extra user ratings with GANs and retrain your model.

## Contributions

We welcome contributions! Feel free to open issues or submit pull requests to help improve this project.

## Team Members

| Name            | Account                                                                    |
| --------------- | -------------------------------------------------------------------------- |
| Mohamed Haytham | @<a href="https://github.com/MedoHaytham" target="_blank">MedoHaytham</a>  |
| Omar Adel       | @<a href="https://github.com/omar55549" target="_blank">omar55549</a>      |
| Mohamed Ahmed   | @<a href="https://github.com/mohamedhanfi" target="_blank">mohamedhanfi</a>|
| Mina Nabil      | @<a href="https://github.com/minasanta" target="_blank">minasanta</a>      |
| Karem Atef      | @<a href="https://github.com/Karem3tef" target="_blank">Karem3tef</a>      |
| Ahmed Monsef    | @<a href="https://github.com/Ahmed1242002" target="_blank">Ahmed1242002</a>|
