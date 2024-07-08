using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NonMaximumSurpression : MonoBehaviour
{

    public Material keepMaterial;
    public Material discardMaterial;
    public float minConfidence;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        MeshCollider[] cellColliders = gameObject.GetComponentsInChildren<MeshCollider>();
        // Loop through pairs of cells
        for (int i = 0; i < cellColliders.Length - 1; i++)
        {
            // Get the current cell collider
            MeshCollider a = cellColliders[i];
            Vector3 positionA = a.gameObject.transform.position;
            Quaternion rotationA = a.gameObject.transform.rotation;
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
                    float aConf = float.Parse(a.transform.parent.name);
                    float bConf = float.Parse(b.transform.parent.name);
                    if (aConf < minConfidence) {
                        Eliminate(a);
                    }
                    if (bConf < minConfidence) {
                        Eliminate(b);
                    }
                    Vector3 dimensionsDifference = a.bounds.size - b.bounds.size;
                    if (Mathf.Abs(dimensionsDifference.x) < 0.5 || Mathf.Abs(dimensionsDifference.y) < 0.5 || Mathf.Abs(dimensionsDifference.z) < 0.5)
                    {
                        if (distance < Mathf.Min(Mathf.Abs(a.bounds.size.x), Mathf.Abs(a.bounds.size.y), Mathf.Abs(a.bounds.size.z))) {
                            if (aConf < bConf) {
                                Eliminate(a);
                            } else {
                                Eliminate(b);
                            }
                        }
                    }
                }
            }
        }  
    }

    void Eliminate(Collider collider) {
        collider.transform.parent.GetComponent<Renderer>().material = discardMaterial;
    }

}
