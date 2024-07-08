using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class RunT8Experiment : MonoBehaviour
{

    public List<TextAsset> sourceFiles;
    public float scalingFactor = 0.1f;
    public float pixelsBetweenPlanes = 21.81f;
    public int numberOfPlanes = 11;
    public int cellMeshLayers = 17;
    public GameObject cellProto;
    public Gradient colorCoding;
    private List<GameObject> embryoContainers = new List<GameObject>();

    // Start is called before the first frame update
    IEnumerator Start()
    {
        // Instiatiate embryos
        for (int i = 0; i < sourceFiles.Count; i++)
        {
            TextAsset file = sourceFiles[i];
            GameObject go = new GameObject($"Embryo-{file.name}");
            embryoContainers.Add(go);
            // Move the object so it is separate
            go.transform.position = Vector3.up * 30 * i;
            // Add generation script
            GenerateEmbryoPoly generator = go.AddComponent<GenerateEmbryoPoly>();
            generator.scalingFactor = scalingFactor;
            generator.pixelsBetweenPlanes = pixelsBetweenPlanes;
            generator.numberOfPlanes = numberOfPlanes;
            generator.cellMeshLayers = cellMeshLayers;
            generator.cellProto = cellProto;
            generator.generateColliders = true;
            generator.sourceFile = file;
            // Visualise the cell contacts
            VisualizeContactGraph graphVisualiser = go.AddComponent<VisualizeContactGraph>();
            graphVisualiser.colorCoding = colorCoding;
            // Add analytics
            go.AddComponent<CellContactAnalyzer>();
            go.AddComponent<CellSizeAnalyzer>();
        }
        // Wait for them to update
        yield return new WaitForSeconds(1);
        // Extract the data we want
        List<KeyValuePair<string, string[]>> metrics = new List<KeyValuePair<string, string[]>>();
        foreach (GameObject embryoContainer in embryoContainers) {
            CellContactAnalyzer cellContactAnalyzer = embryoContainer.GetComponent<CellContactAnalyzer>();
            CellSizeAnalyzer cellSizeAnalyzer = embryoContainer.GetComponent<CellSizeAnalyzer>();
            try {
                float averageNeighbors = cellContactAnalyzer.GetAverageNeighbors();
                float maxNeighbors = cellContactAnalyzer.GetMaxNeighbors();
                float minNeighbors = cellContactAnalyzer.GetMinNeighbors();
                // T4Shape t4Shape = analyzer.GetT4Shape();
                if (cellContactAnalyzer.transform.childCount != 8) {
                    throw new IncorrectNumberOfCellsException();
                }
                metrics.Add(
                    new KeyValuePair<string,string[]>(
                        cellContactAnalyzer.gameObject.name,
                        new string[] {
                            cellContactAnalyzer.GetAdjacencyMatrixString(),
                            cellSizeAnalyzer.GetCellVolumeString()
                        }
                    )
                );
            } catch (IncorrectNumberOfCellsException) {
                // Ignore this embryo if not the right number of cells
                Destroy(cellContactAnalyzer.gameObject);
                continue;
            }
            // Destroy(analyzer.gameObject);
        }
        // Save the data to a file
        File.WriteAllLines(
            "T8-Output.csv",
            metrics.Select(x => $"{x.Key}, {x.Value[0]}, {x.Value[1]}")
        );
    }

}
