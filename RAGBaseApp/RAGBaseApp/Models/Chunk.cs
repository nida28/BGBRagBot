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
        public string Text { get; set; }                // Cleaned BGB paragraph
        public float[] Embedding { get; set; }          // Vector from OpenAI
    }

}
