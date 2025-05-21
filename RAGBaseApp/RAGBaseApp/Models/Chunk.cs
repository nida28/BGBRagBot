using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RAGBaseApp.Models
{
    public class Chunk
    {
     public string Id { get; set; }  
     public string SectionId { get; set; }
     public string Text { get; set; }
     public string SectionNumber { get; set; }  
     public string SectionTitle { get; set; }   
     public float[] Embedding { get; set; }        
    }

}
