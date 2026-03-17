import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.load_dataset import load_dataset
from src.pca_feature_extraction import compute_pca
from src.eigenfaces import generate_signatures, project_image, display_eigenfaces
from src.ann_classifier import train_test_split_signatures, train_ann, predict_ann, evaluate_accuracy
from src.test import predict_identity, visualize_prediction, get_reference_image

def main():
    dataset_path = "dataset/faces"
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    print("1. Loading Dataset...")
    X, y, label_map = load_dataset(dataset_path, image_size=(100, 100))
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    # We will test different values of k
    k_values = [10, 20, 30, 40, 50]
    accuracies = []
    
    best_k = None
    best_acc = 0
    best_models = None
    
    print("\n2. Evaluating PCA + ANN for different values of k...")
    for k in k_values:
        print(f"\n--- Testing with k = {k} ---")
        
        # PCA
        mean_face, eigenvectors, eigenvalues, A = compute_pca(X, k=k)
        
        # Generate Signatures
        signatures = generate_signatures(A, eigenvectors)
        
        # Split Data (consistent split using random_state)
        X_train, X_test, y_train, y_test = train_test_split_signatures(signatures, y, test_size=0.4, random_state=42)
        
        # Train ANN
        classifier = train_ann(X_train, y_train, hidden_layer_sizes=(100,), max_iter=1000)
        
        # Predict and Evaluate
        predictions, probabilities = predict_ann(classifier, X_test)
        acc = evaluate_accuracy(y_test, predictions)
        print(f"Accuracy for k={k}: {acc * 100:.2f}%")
        
        accuracies.append(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_models = {
                'mean_face': mean_face,
                'eigenvectors': eigenvectors,
                'classifier': classifier,
                'label_map': label_map,
                'image_size': (100, 100)
            }

    print(f"\nBest accuracy achieved with k = {best_k} ({best_acc * 100:.2f}%)")
    
    # Plot accuracy vs k
    print("\n3. Plotting Accuracy vs k graph...")
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title("Classification Accuracy vs Number of Eigenvectors (k)")
    plt.xlabel("Number of Eigenvectors (k)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    graph_path = os.path.join(results_dir, "accuracy_graph.png")
    plt.savefig(graph_path)
    print(f"Graph saved to {graph_path}")
    # plt.show()

    # Display best eigenfaces
    print("\n4. Displaying top Eigenfaces for best model...")
    eigenface_path = os.path.join(results_dir, "best_eigenfaces.png")
    display_eigenfaces(best_models['eigenvectors'], image_size=(100, 100), num_faces=10, save_path=eigenface_path)
    
    # Face Matching and Reconstruction Test
    print("\n5. Running Face Matching and Reconstruction Test...")
    # Test on an example image from the dataset
    test_face_path = "dataset/faces/Aamir/face_101.jpg"
    
    if os.path.exists(test_face_path):
        name, conf, img, reconstructed = predict_identity(test_face_path, best_models, confidence_threshold=0.4)
        print(f"Tested Image ({test_face_path}):")
        print(f"Prediction: {name} (Confidence: {conf:.4f})")
        
        # Get a reference image of the predicted person (for "Original Face" requirement)
        reference_img = get_reference_image(name, dataset_path="dataset/faces")
        
        test_result_path = os.path.join(results_dir, "face_match_reconstruction.png")
        visualize_prediction(img, name, conf, 
                             reconstructed_img=reconstructed, 
                             reference_img=reference_img, 
                             save_path=test_result_path)
    else:
        print(f"Test image {test_face_path} not found.")

if __name__ == "__main__":
    main()
