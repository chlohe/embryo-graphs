using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using TMPro;

public class CellContactAnalyzer : MonoBehaviour
{
    public VisualizeContactGraph visualizer;
    public TextMeshProUGUI outputTextMesh;
    private float averageNeighbors;
    private int minNeighbors;
    private int maxNeighbors;

    // Start is called before the first frame update
    void Start()
    {
        if (!visualizer)
        {
            visualizer = GetComponent<VisualizeContactGraph>();
        }
    }

    // Update is called once per frame
    void Update()
    {
        int[] neighbors = CalculateNeighbors();
        averageNeighbors = neighbors.Length > 0 ? (float)neighbors.Average() : 0;
        minNeighbors = neighbors.Length > 0 ? neighbors.Min() : 0;
        maxNeighbors = neighbors.Length > 0 ? neighbors.Max() : 0;
        // Display results
        if (outputTextMesh)
        {
            outputTextMesh.text = $"Average Neighbors: {neighbors.Average():f4}";
        }
    }

    int[] CalculateNeighbors() {
        float[,] adj = visualizer.GetAdjacencyMatrix();
        if (adj != null)
        {
            // Compute number of neighbors per cell
            int[] neighbors = new int[adj.GetLength(0)];
            for (int i = 0; i < adj.GetLength(0); i++)
            {
                for (int j = 0; j < adj.GetLength(1); j++)
                {
                    neighbors[i] += adj[i, j] > 0 ? 1 : 0;
                }
            }
            return neighbors;
        }
        return new int[] {};
    }

    public string GetAdjacencyMatrixString() {
        return MatrixToString(visualizer.GetAdjacencyMatrix(), "/");
    }

    string MatrixToString<T>(T[,] matrix, string newlineDelimiter = "\n")
    {
        // Print a 2D matrix
        string output = "";
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            for (int j = 0; j < matrix.GetLength(1); j++)
            {
                output += " " + matrix[i, j].ToString();
            }
            output += newlineDelimiter;
        }
        return output;
    }

    public float GetAverageNeighbors() {
        return averageNeighbors;
    }

    public float GetMinNeighbors() {
        return minNeighbors;
    }

    public float GetMaxNeighbors() {
        return maxNeighbors;
    }

    public T4Shape GetT4Shape() {
        int[] neighbors = CalculateNeighbors();
        // Check if we are even dealing with a 4 cell embryo
        if (neighbors.GetLength(0) != 4) {
            throw new IncorrectNumberOfCellsException(4, neighbors.GetLength(0));
        }
        // Count frequency of each number of contacts
        int[] frequency = new int[4];
        foreach (int neighborCount in neighbors) {
            frequency[neighborCount] += 1;    
        }
        // Categorise them
        if (frequency[3] == 4) {
            if (frequency[1] == 3) {
                return T4Shape.OpenY;
            } else {
                return T4Shape.Tetrahedral;
            }
        }
        if (frequency[2] == 2) {
            if (frequency[3] == 2) {
                return T4Shape.Pseudotetrahedral;
            } else if (frequency[1] == 2) {
                return T4Shape.Linear;
            } else {
                return T4Shape.ClosedY;
            }
        }
        if (frequency[2] == 4) {
            return T4Shape.Planar;
        }
        return T4Shape.Other;
    }

}
