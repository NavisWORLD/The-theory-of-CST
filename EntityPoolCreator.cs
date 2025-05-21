using UnityEditor;
     using UnityEngine;

     public class EntityPoolCreator
     {
         [MenuItem("Tools/Create Entity Pool")]
         static void CreatePool()
         {
             GameObject cosmicEngine = GameObject.Find("CosmicEngine");
             if (!cosmicEngine) cosmicEngine = new GameObject("CosmicEngine");
             int poolSize = 1000;
             for (int i = 0; i < poolSize; i++)
             {
                 GameObject entity = new GameObject($"Entity_{i}");
                 entity.transform.SetParent(cosmicEngine.transform);
                 entity.AddComponent<MeshFilter>();
                 entity.AddComponent<MeshRenderer>();
                 entity.AddComponent<CSTEntityData>();
                 entity.SetActive(false);
             }
             Debug.Log($"Created {poolSize} entities under CosmicEngine");
         }
     }