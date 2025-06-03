# Performance Tables for Transformer Models and Fine-Tuning

## Table 1: Transformer Model with Mapping Network (Split by ID)
| Model                          | Test Cosine Loss | Test Cosine Similarity |
|--------------------------------|------------------|------------------------|
| BAAI/bge-large-en-v1.5         | 0.7926           | 0.2152                 |
| intfloat/e5-large-v2           | 0.8214           | 0.1894                 |

## Table 2: Transformer Model with Mapping Network (Split by Query)
| Model                          | Test Cosine Loss | Test Cosine Similarity |
|--------------------------------|------------------|------------------------|
| BAAI/bge-large-en-v1.5         | 0.5429           | 0.4671                |
| intfloat/e5-large-v2           | 0.5890           | 0.4125                 |

## Table 3: Direct Fine-Tuning (Split by Query)
| Model                          | Test Cosine Loss | Test Cosine Similarity |
|--------------------------------|------------------|------------------------|
| BAAI/bge-large-en-v1.5         | 0.0451           | 0.5265                 |
| intfloat/e5-large-v2           | 0.0684           | 0.4507                 |

## Table 4: Fine-Tuning (Split by ID)
| Model                          | Test Cosine Loss | Test Cosine Similarity |
|--------------------------------|------------------|------------------------|
| BAAI/bge-large-en-v1.5         | 0.0711           | 0.1182                 |
| intfloat/e5-large-v2           | 0.0665           | 0.1233                 |