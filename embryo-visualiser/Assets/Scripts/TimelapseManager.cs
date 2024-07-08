using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using UnityEngine;
using System.Globalization;
using System;

public class TimelapseManager : MonoBehaviour
{
    public float scalingFactor = 0.1f;
    public float pixelsBetweenPlanes = 1.5f;
    public int numberOfPlanes = 11;
    public int cellMeshLayers = 7;
    public GameObject cellProto;
    public bool generateColliders = false;
    public float colliderTolerance = 0.05f;
    private List<Mesh> cellMeshes = new List<Mesh>();
    private List<GameObject> cellObjects = new List<GameObject>();
    private Dictionary<float, Transform> steps = new Dictionary<float, Transform>();
    [Range(0, 70)]
    public float t;

    public void InitializeEmbryo(TextAsset sourceFile)
    {
        // Get rid of the current embryo
        foreach (Transform child in transform.GetComponentInChildren<Transform>())
        {
            Destroy(child.gameObject);
            steps.Clear();
            cellMeshes.Clear();
        }
        // Reset container rotation
        transform.rotation = Quaternion.identity;
        // Parse the new embryo data
        string text = sourceFile.text;
        string[] lines = text.Split('\n');
        Regex regex = new Regex(@"\([0-9]*,\s*[0-9]*\)");
        float currentTimestamp = -1;
        foreach (string line in lines)
        {
            if (line.Trim().Length == 0)
            {
                continue;
            }
            // Check if we are dealing with a new timestep
            if (line.Contains("=="))
            {
                string timestampString = line.Replace("=", "").Trim();
                float.TryParse(timestampString, NumberStyles.Any, CultureInfo.InvariantCulture, out currentTimestamp);
                // Create a gameobject to contain the cells
                Transform container = new GameObject(currentTimestamp.ToString()).transform;
                container.parent = transform;
                steps.Add(currentTimestamp, container);
                continue;
            }
            // Make sure we have a timestamp
            if (currentTimestamp == -1f)
            {
                Debug.LogError("Invalid timestamps!");
            }
            // Parse out center coords
            Vector3 center = Vector3.zero;
            float.TryParse(line.Split(' ')[0], out center.x);
            float.TryParse(line.Split(' ')[1], out center.z);
            float.TryParse(line.Split(' ')[2], out float depth);
            float.TryParse(line.Split(' ')[3], out float confidence);
            // Get vert coords
            List<Vector3> coords = new List<Vector3>();
            Match match = regex.Match(line);
            while (match.Success)
            {
                string[] coordStrings = match.Value.Replace("(", "").Replace(")", "").Split(',');
                float.TryParse(coordStrings[0], out float x);
                float.TryParse(coordStrings[1], out float z);
                coords.Add(new Vector3(x, 0, z));
                match = match.NextMatch();
            }
            // Scale coords
            coords = coords.Select(x => x * scalingFactor).ToList();
            center *= scalingFactor;
            // Generate mesh
            GenerateMesh(coords, center, pixelsBetweenPlanes * scalingFactor, depth, numberOfPlanes, currentTimestamp, confidence);
        }
        CenterMeshes();
        // Add colliders if necessary
        if (generateColliders)
        {
            AddColliders();
        }
        // Rotate to give a "top-down" view of the embryo akin to the microsope
        transform.Rotate(new Vector3(90, 0, 0));
    }

    // Update is called once per frame
    void Update()
    {
        if (steps.Count > 0)
        {
            // Find closest step to current time
            Transform closestStepContainer = steps.OrderBy(x => Mathf.Abs(x.Key - t)).ToList()[0].Value;
            foreach (Transform container in steps.Values)
            {
                if (container == closestStepContainer)
                {
                    container.gameObject.SetActive(true);
                }
                else
                {
                    container.gameObject.SetActive(false);
                }
            }
        }
    }

    void GenerateMesh(List<Vector3> coords, Vector3 center, float interPlaneDistance, float depth, int planeCount, float currentTimestamp, float confidence)
    {
        Transform container = steps[currentTimestamp];
        GameObject cell = Instantiate(cellProto,
            transform.position + Vector3.up * interPlaneDistance * depth - Vector3.up * planeCount,
            Quaternion.identity, container
        ) as GameObject;
        if (confidence > 0) {
            cell.name = confidence.ToString();
        }
        cellObjects.Add(cell);
        Mesh mesh = cell.GetComponent<MeshFilter>().mesh;
        cellMeshes.Add(mesh);
        // Calculate radius
        float xMax = coords.Max(v => v.x);
        float xMin = coords.Min(v => v.x);
        float zMax = coords.Max(v => v.z);
        float zMin = coords.Min(v => v.z);
        float equatorialDiameter = (xMax - xMin + zMax - zMin) / 2;
        float distanceBetweenLayers = equatorialDiameter / cellMeshLayers;

        // Generate layer verts
        List<List<Vector3>> layers = new List<List<Vector3>>();
        List<int> vertLayerIndex = new List<int>();
        vertLayerIndex.Add(0);
        for (int i = -Mathf.FloorToInt((float)cellMeshLayers / 2); i <= Mathf.FloorToInt((float)cellMeshLayers / 2); i++)
        {
            List<Vector3> layer = ShrinkLayer(coords, center, Mathf.Sqrt(1f - Mathf.Pow(Mathf.Abs(2 * i / (float)(cellMeshLayers - 1)), 2)))
                .Select(v => new Vector3(v.x, i * distanceBetweenLayers, v.z))
                .ToList();
            layers.Add(layer);
            vertLayerIndex.Add(vertLayerIndex.Last() + layer.Count);
        }
        Vector3[] verts = layers.SelectMany(x => x).ToArray();
        mesh.vertices = verts;

        // Generate triangles to link layers together
        int[] triangles = new int[verts.Length * 3 * 2];

        for (int i = 0; i < layers.Count - 1; i++)
        {
            // Make triangles between pairs of adjacent layers
            int layerBaseLower = vertLayerIndex[i];
            int layerBaseUpper = vertLayerIndex[i + 1];
            for (int j = 0; j < layers[i].Count - 1; j++)
            {
                triangles[6 * (layerBaseLower + j)] = layerBaseLower + j;
                triangles[6 * (layerBaseLower + j) + 1] = layerBaseLower + j + 1;
                triangles[6 * (layerBaseLower + j) + 2] = layerBaseUpper + j;
                triangles[6 * (layerBaseLower + j) + 3] = layerBaseLower + j + 1;
                triangles[6 * (layerBaseLower + j) + 4] = layerBaseUpper + j + 1;
                triangles[6 * (layerBaseLower + j) + 5] = layerBaseUpper + j;
            }
            // Close the loop
            int k = layers[i].Count - 1;
            triangles[6 * (layerBaseLower + k)] = layerBaseLower + k;
            triangles[6 * (layerBaseLower + k) + 1] = layerBaseLower;
            triangles[6 * (layerBaseLower + k) + 2] = vertLayerIndex[i + 2] - 1;
            triangles[6 * (layerBaseLower + k) + 3] = layerBaseLower;
            triangles[6 * (layerBaseLower + k) + 4] = layerBaseUpper;
            triangles[6 * (layerBaseLower + k) + 5] = vertLayerIndex[i + 2] - 1;
        }

        // Assign triangles
        mesh.triangles = triangles;
        mesh.RecalculateBounds();

        // Generate UVs (causes WebGL to crash on iOS so don't enable)
        // Vector2[] uv = verts.Select(v => new Vector2(1 / xMax, 1 / zMax)).ToArray();
        // mesh.uv = uv;
        mesh.RecalculateNormals();
    }

    List<Vector3> ShrinkLayer(List<Vector3> coords, Vector3 center, float factor)
    {
        return coords.Select(x => ((x - center) * factor) + center).ToList();
    }

    void CenterMeshes()
    {
        // Freeload off the physics engine to get the embryo center
        Vector3 center = Vector3.zero;
        foreach (Mesh mesh in cellMeshes)
        {
            Bounds bounds = mesh.bounds;
            center += bounds.center;
        }
        center /= cellMeshes.Count;
        foreach (Mesh mesh in cellMeshes)
        {
            mesh.vertices = mesh.vertices.Select(v => v - center).ToArray();
            mesh.RecalculateBounds();
        }
    }

    public Vector2 GetTimeBounds()
    {
        return new Vector2(
            steps.Count > 0 ? steps.Keys.Min() : 0,
            steps.Count > 0 ? steps.Keys.Max() : 0
        );
    }

    void AddColliders()
    {
        foreach (GameObject cell in cellObjects)
        {
            // Create object to contain collider (separate so we can scale it)
            GameObject colliderContainer = new GameObject("Collider");
            colliderContainer.transform.parent = cell.transform;
            colliderContainer.transform.localPosition = Vector3.zero;
            // Add the collider
            MeshCollider collider = colliderContainer.AddComponent<MeshCollider>();
            collider.sharedMesh = cell.GetComponent<MeshFilter>().mesh;
            collider.convex = true;
            // Scale the collider
            float scaleFactor = 1 + colliderTolerance;
            colliderContainer.transform.localScale = Vector3.one * scaleFactor;
            // Center the container
            Bounds bounds = collider.bounds;
            Vector3 displacement = colliderContainer.transform.position - bounds.center;
            colliderContainer.transform.position = bounds.center + displacement * scaleFactor;
        }
    }
}