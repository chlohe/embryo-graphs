using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class SimpleEmbryoViewer : MonoBehaviour
{

    public TextAsset embryo;
    public Material edgeMaterial;
    public Gradient colorCoding;
    public TextMeshProUGUI outputTextMesh;
    private TimelapseManager manager;

    // Start is called before the first frame update
    void Start()
    {
        // Create embryo
        manager = GetComponent<TimelapseManager>();
        manager.InitializeEmbryo(embryo);
        // Add graph visualisation to all steps
        Transform[] steps = transform.GetComponentsInChildren<Transform>();
        foreach (Transform step in steps)
        {
            if (step.parent == transform) {
                // Only look at our immediate children
                VisualizeContactGraph visualizer = step.gameObject.AddComponent<VisualizeContactGraph>();
                visualizer.edgeMaterial = edgeMaterial;
                visualizer.colorCoding = colorCoding;
                CellContactAnalyzer analytics = step.gameObject.AddComponent<CellContactAnalyzer>();
                analytics.outputTextMesh = outputTextMesh;
            }
        }
    }

    // Update is called once per frame
    void Update()
    {

    }
}
