using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GenerateEmbryoEllipsoid : MonoBehaviour
{

    public TextAsset sourceFile;
    public GameObject cellProto;

    public List<EllipsoidCell> cells = new List<EllipsoidCell>();
    // Start is called before the first frame update
    void Start()
    {
        float scaleFactor = 21.8f;
        string text = sourceFile.text;
        string[] lines = text.Split('\n');
        foreach (string line in lines)
        {
            if (line.Length < 1)
            {
                continue;
            }
            // Parse out the params
            string[] splitText = line.Split(' ');
            EllipsoidCell cell = new EllipsoidCell();
            int.TryParse(splitText[0], out cell.cx);
            int.TryParse(splitText[1], out cell.cy);
            int.TryParse(splitText[2], out cell.w);
            int.TryParse(splitText[3], out cell.h);
            int.TryParse(splitText[4], out cell.angle);
            int.TryParse(splitText[5], out cell.depth);
            cells.Add(cell);
        }
        // Move own transform to center of embryo
        Vector3 embryoCenterPos = new Vector3();
        foreach (EllipsoidCell cell in cells)
        {
            embryoCenterPos += new Vector3((float)(cell.cx / scaleFactor), (float)(cell.depth - 5), (float)(cell.cy / scaleFactor));
        }
        transform.position = embryoCenterPos / cells.Count;
        // Place cells
        foreach (EllipsoidCell cell in cells)
        {
            Vector3 pos = new Vector3((float)(cell.cx / scaleFactor), (float)(cell.depth - 5), (float)(cell.cy / scaleFactor));
            Vector3 rot = new Vector3(0, cell.angle + 90, 0);
            Vector3 scale = new Vector3((float)(1.1 * cell.h / scaleFactor), (float)(1.1 * (cell.w + cell.h) / (2 * scaleFactor)), (float)(1.1 * cell.w / scaleFactor));
            GameObject go = Instantiate(cellProto, pos, Quaternion.Euler(rot), transform) as GameObject;
            go.transform.localScale = scale;
        }
    }

    // Update is called once per frame
    void Update()
    {

    }
}

[System.Serializable]
public class EllipsoidCell
{

    public int cx, cy, w, h, angle, depth;
    public EllipsoidCell() { }

}