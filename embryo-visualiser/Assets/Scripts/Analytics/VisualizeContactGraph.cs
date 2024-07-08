using System.Collections.Generic;
using UnityEngine;

public class VisualizeContactGraph : MonoBehaviour
{
    public Material edgeMaterial;
    public float edgeWidth = 0.3f;
    public Gradient colorCoding;
    public bool onlyUpdateAtStart = true;
    private float[,] adjacencyMatrix;

    // Start is called before the first frame update
    void Start()
    {
        if (onlyUpdateAtStart)
        {
            UpdateVisualization();
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (!onlyUpdateAtStart)
        {
            UpdateVisualization();
        }
    }

    void UpdateVisualization()
    {
        MeshCollider[] cellColliders = gameObject.GetComponentsInChildren<MeshCollider>();
        adjacencyMatrix = new float[cellColliders.Length, cellColliders.Length];
        // Loop through pairs of cells
        for (int i = 0; i < cellColliders.Length - 1; i++)
        {
            // Get the current cell collider
            MeshCollider a = cellColliders[i];
            Vector3 positionA = a.gameObject.transform.position;
            Quaternion rotationA = a.gameObject.transform.rotation;
            // Hunt down a LineRenderer so that we can render lines
            LineRenderer[] lines = a.gameObject.GetComponentsInChildren<LineRenderer>();
            foreach (LineRenderer line in lines)
            {
                Destroy(line.gameObject);
            }
            // Init places to store positions / colors
            for (int j = i + 1; j < cellColliders.Length; j++)
            {
                MeshCollider b = cellColliders[j];
                Vector3 positionB = b.gameObject.transform.position;
                Quaternion rotationB = b.gameObject.transform.rotation;

                Vector3 direction;
                float distance;
                // Check if the two cells overlap
                bool overlapped = Physics.ComputePenetration(
                    a, positionA, rotationA,
                    b, positionB, rotationB,
                    out direction, out distance
                );
                // If so, draw a line between them
                Color segmentColor = Random.ColorHSV();
                if (overlapped)
                {
                    // Create line
                    GameObject lineContainer = new GameObject("Line");
                    lineContainer.transform.parent = a.transform;
                    LineRenderer line = lineContainer.AddComponent<LineRenderer>();
                    line.material = edgeMaterial;
                    line.startWidth = edgeWidth;
                    line.endWidth = edgeWidth;
                    line.positionCount = 2;
                    // Set endpoints to cell centers
                    line.SetPositions(new Vector3[] {
                        a.bounds.center,
                        b.bounds.center
                    });
                    line.useWorldSpace = false;
                    // Set color
                    line.startColor = colorCoding.Evaluate(distance);
                    line.endColor = colorCoding.Evaluate(distance);
                    // Update adjacency matrix
                    adjacencyMatrix[i, j] = distance;
                    adjacencyMatrix[j, i] = distance;
                }
            }
        }
    }

    public float[,] GetAdjacencyMatrix() {
        return adjacencyMatrix;
    }

    string MatrixToString<T>(T[,] matrix)
    {
        // Print a 2D matrix
        string output = "";
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            for (int j = 0; j < matrix.GetLength(1); j++)
            {
                output += " " + matrix[i, j].ToString();
            }
            output += '\n';
        }
        return output;
    }

}
