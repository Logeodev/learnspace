## REST API requests exemples

* Create an index
```js
POST /index_name
```

* Insert document in index
```js
PUT /index_name/_doc/$id
{
  "key1" : value1,
  "key2" : value2
}
```
> `id` parameter is optional, is generated if missing.

Then we can retrieve it with `GET /index_name/_doc/$id`.

* Updating document
```js
POST /index_name/_update/$id
{
  "doc": {
     "some_key" : value,
     "tags" : ["tag1", "tag2"]
  }
}
```
> If the key updated exists the value will be modified, if the key doesn't exist in the doc then it will be added to the object.

* Updating document by operation
```js
POST /index_name/_update/$id
{
  "script" : {
    "source" : """
      if (ctx._source.some_key == 0) {
        ctx.op = 'noop';
      }
      ctx._source.some_key--
    """
  }
}
```
>`ctx` = context, meaning what doc the request is about, `noop` = no operation and will affect the result status, and `--` means `val = val -1`

> additional scripting will come later

* Update many documents at a time (WHERE clause)
```js
POST /index_name/_update_by_query
{
  "conflicts" : "proceed",
  "script" : {
    "source" : "ctx._source.some_key++"
  },
  "query" : {
    "match_all" : {}
  }
}
```
> `conflicts` value will give how the request should manage conflicts on some documents of the query.

> `_delete_by_query` parameter delete some documents depending on the matching.

* Batch processing (BULK API)
Indexing many docs at a time :
```js
POST /_bulk
{
  "index": {
    "_index": "index_name",
    "_id": id
  }
}
{
  "key1": "value1",
  "key2": "value2"
}
{
  "create": {
    "_index": "index_name",
    "_id": id
  }
}
{
  "key1": "value3",
  "key2": "value4"
}
```
> With `POST /_bulk` we specify each action then the object of the action, so we can do many creation/deleting/update we want

If we specify `POST /index_name/_bulk` there's no need to indicate the index in which we're working each time.

> If one action fails, the others will still be done !

* From cURL

Simple request : 
```
curl -XGET "http://localhost:9200/kibana_sample_data_flights/_search?q=*"
```

Bulk request :
```
curl -H "Content-Type: application/x-ndjson" -XPOST http://localhost:9200/index_name/_bulk 
  --data-binary "@file-name.json"
```

## Analyzers

* standard analyser = split on spaces (=tokenizer) + lowercase (=token filter)
* others are [already built](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-analyzers.html)
* possibility to build some ourselves

```json
POST /_analyse
{
  "text": "Here can BE ANY text !",
  "analyser": "standard"
}
```
> will return a list of tokens which are the words and characters set to lowercase.

Analyzers are used in order to make a search in the indexes and documents ! Inverted indices tables (stored in Apache Lucene) are created from the analyse, which allows searching for what doc has some word.

### Stemming analyzers

Used in any search queries or even when storing a string, depends on a language (see [language analyzers](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-lang-analyzer.html)).
:::mermaid
graph TB;
I(I loved drinking bottles of Coke on last year's vacation.);
B(I love drink bottl of Coke on last year vacat.);
E(i, love, drink, bottl, coke, last, year, vacat);
I --stemming--> B;
B --removing stop words--> E;
:::
It removes any information that's not useful information to search. What is removed is tense markers, plurals or coordination words, which are adding context into a sentence but not really any sense.

### Custom analyzer

Exemple of an analyzer with some html string as input :
```json
PUT /analyzer_name
{
  "settings": {
    "analysis": {
      "filter": {
        "danish_stop": {"type": "stop", "stopwords": "_danish_"}
      },
      "analyzer": {
        "my_analyzer": {
          "type": "custom",
          "char_filter": ["html_strip"],
          "tokenizer": "standard",
          "filter": ["lowercase", "danish_stop", "asciifolding"]
        }
      }
    }
  }
}
```
Then we can use it with `POST /analyzer_name/_analyse {"analyzer": "my_analyzer", "text": "some_text_to_analyse"}`. It will remove the Danish stop words, the HTML tags, and transform any special character to the ASCII norm.

> Can setup many analyzers in one query !

## Mapping

= create a link between some elements

For exemple, property mapping is assigning a type to a value (text, boolean, ...), this is done automatically when indexing (= data coercion, which can be disabled).

Create a property mapping is used to specify what values should be in a document of some index :
```json
PUT /index_name
{
  "mappings": {
    "properties": {
      "key1": {"type": "float"},
      "key2": {"type": "integer"},
      "key3": {
        "properties": {
          "key4": {"type": "text"},
          "key5": {"type": "keyword"}
        }
      }
    }
  }
}
```
Then, when putting a document in this index, ES will try to coerce all given values to the mapping ! If given doc values can't be coerced, then indexing will return an error.

> Dynamic Mapping happens when we don't specify a mapping, as previously. Can get the mapping of a given index with : `GET /index_name/_mapping`. Can get the mapping of a specific field of an object with `GET /index_name/_mapping/field/some_key`. We can set the dynamic mapping to `false` or `"strict"`in some cases, i.e. strict will not allow not mapped fields to be added.

In spite of nested mapping, we can declare fields like `"key3.key4: {"type": "text"}"` in the previous exemple.

Afterwards (by default, all fields are optional in a doc), we can add fields with mapping (but usually can't modify an existing field mapping) with :
```json
PUT /index_name/_mapping
{
  "properties": {
    "date_creation": {"type": "date", "format": "dd/MM/yy"}
  }
}
```
> A date field will by default (else wise specify the format) be converted into a _long_ number, the timestamp. When searching on dates, this conversion also happens (tho the result shown will be as it was stored). 

:warning:There are many [other property parameters](https://www.elastic.co/guide/en/elasticsearch/reference/current/properties.html), like disable NULL values, specifications to save storage space or __setting an alias for a key or index__.

If we need to modify an existing mapping, the easiest would be to copy docs to a new index with the new mapping :
```json
POST /_reindex
{
  "source": {"index": "old_index"},
  "dest": {"index": "new_index"}
}
```
Then, `DELETE /old_index/_delete_by_query {"query": {"match_all": {}}}`.
> Can add scripts in the `_reindex` API, like doing something when one value is `null`. Or add a matching query on some values ! (So it can be used when some fields are useless with time, and we want to remove them in the new version.)

> Can use the `/_template` API to create __index templates__.


## Searching

* Quick search :
```http
GET /indexName/_search?q=fieldName:value
```
> `q=*` to select every items. Logic can be added with `q=fiedl1:value1 AND field2:value2` and so on.

* Score :
Depends on 
1. Term Frequency (in the field requested) = how many times the term appears in the index
2. Inverse Term Frequency (in the indexes) = the more a term appears in an index the lower its score
3. algorithm Okapi BM25 = handles stop words and [stemming](https://dev.azure.com/SDEA/Data%20Science/_wiki/wikis/Data-Science.wiki?wikiVersion=GBwikiMaster&pagePath=/User%20guide%20ElasticSearch&pageId=680&_a=edit#stemming-analyzers), plus field-length norm factor

> It is possible to change the way ES calculates a score.

With a query like `GET GET /indexName/_search?explain {...}`, ES gives the way it has calculated the scores of returned documents.

* Term level query :
```js
{
  "query": {
    "term": {
      "fieldName": value
    }
  }
}
```
Searches for exact values with ITF (standard analyser). Note that fields passed through the standard analyzer and will be lowercased here. So the value given won't match if its first letter is uppercase (the value given in the query won't go with std analyzer with term level queries).
Better use would be to use term level queries on numeric values only. Or use it to find specific words in list field, like tags.

* Get documents by IDs :
```js
GET kibana_sample_data_flights/_search
{
  "query": {
    "ids": {
      "values": ["6-oJuoUB5d0yRLsiRcQf"]
    }
  }
}
```

* Documents having value in some range :
```js
GET kibana_sample_data_flights/_search
{
  "query": {
    "range": {
      "AvgTicketPrice": {
        "gte": 0, // greater than or equals
        "lte": 150 // lower than or equals
      }
    }
  }
}
```
> Works also with dates `yyyy/MM/dd` (default format). To use a different format, specify it along `gte` and `lte`, with `"format" : "dd-MM-yyyy"` for exemple.

> _Relative dates_ : `"2020/01/01||-1y/M"` will designate the 1st January of 2018 (rounded by month). Can also use `"now"` to get today's date.

* Non null values, with the `"exists"` key :
To retrieve the documents in which a specified field is not null.
```js
GET indexName/_search
{
  "query": {
    "exists": {
      "field": "fieldName"
    } 
  }
}
```

* Start with, in text queries :
```js
{
  "query": {
    "prefix": {
      "FIELD": "value"
    }
  }
}
```

* Wildcards : `*` or `?`
`wil*rd` would match with any words starting with `wil` and ending with `rd`.
`wildc?rd` would match with any words where we can change `?` with another single character.

* Regex can be used with the `"regexp"` key word in the query.

* Full text query :
```js
{
  "query": {
    "match": {
      "fieldName": value
    }
  }
}
```
Searching if the value given appears in a document, the value will first go through __the standard analyzer___, meaning here fields won't be case sensitive.
Can give a full sentence like `recipe with pasta or spaghetti` can retrieve with scores all the recipes even if those have spaghettis, pasta or both (because the given sentence will go through the std analyzer).
Can also add the `"operator"` key so we can specify the logical operator to use in the matching.
Can use `"match_phrase"` instead of `"match"` to retrieve only if the doc has the field written in the same order than the value given.

* Search on many fields :
```js
Get recipes/_search
{
  "query": {
    "multi_match": {
      "query": "valueToSearch",
      "fields": ["field1", "field2"]
    }
  }
}
```

* Boolean Logic to make search queries :

Retrive documents with certain text in some field AND having values between a range on some other field would be
```js
Get recipes/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "FIELD": "TEXT"
          }
        }, 
        {
          "range": {
            "FIELD": {
              "gte": 10,
              "lte": 20
            }
          }
        }
      ]
    }
  }
}
```

A boolean query can take `boost`, `filter`, `must`, `must_not`, `should` (= better score if in document) and `minimum_should_match` arguments, which acts like the logical operators on the given matching patterns of a list.

We can use as many of these operators in a single boolean query. As well as many matching patterns inside of them (exemple above, `must` has 2 matching patterns in).

We can also name an operation :
```js
Get recipes/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "title": {
              "query": "Beans",
              "_name": "must_have_beans"
            }
          }
        }
      ]
    }
  }
}
```

We can translate `match` queries as boolean queries (they are just simplified). Using `"operator"="and"` is a `must` on the terms of the given value, `"operator"="or"` is a `should`

* Query on nested objects :

```js
Get recipes/_search?explain
{
  "query": {
    "nested": {
      "path": "path_to_nested_doc",
      "inner_hits": {},
      "query": {}
    }
  }
}
```
For exemple, if we have different docs being the departments of a company, and listing its employees, we should use the nested type query (where the `path` would be the employees list). Specifying `inner_hits` will retrieve only the sub elements that match the query (and not the whole doc containing this element).

## Relational database

Elastic Search is not a relational database, hence we don't have any foreign keys or equivalent.
Tho, we can create a mapping property to simulate a link between documents :
```js
PUT /indexName
{
  "mappings": {
    "_doc": {
      "properties": {
        "some_join_name": {
          "type": "join",
          "relations": {
            "parentName": "childName"
          }
        }
      }
    }
  }
}
``` 
Then we can create our new indexes of Departments and Employees adding the property just created, with `"some_join_name":"parentName"`. In the case of the children documents :
```js
PUT /indexName/_doc/:id?routing=IDofParentDoc
{
  "key1": value1,
  "key2": value2,
  "some_join_name": {
    "name": "childName",
    "parent": IDofParentDoc
  }
}
```
> In the case of having a parents AND grandparents nested structure, the `routing` argument should always be the higher level (meaning the grandparent) tho the parent's id is the parent.

Now to get the child elements of some doc, use :
```js
GET /recipes/_search
{
  "query": {
    "parent_id": {
      "type": "childName",
      "id": IDofParentDoc
    }
  }
}
```
Or, with more complex query possibilities :
```js
GET /recipes/_search
{
  "query": {
    "has_parent": {
      "parent_type": "parentName",
      "score_mode": true, // enable the scores
      "query": {
        // Any query on the parent can go here
      }
    }
  }
}
```

And to retrieve the parents from a doc matching some query :
```js
GET /recipes/_search
{
  "query": {
    "has_child": {
      "child_type": "childName",
      "query": {
        // Any query on child can go here
      }
    }
  }
}
```

> Score modes :

| Mode | Description |
|:-:|:--|
| min | the lowest score of matching child doc is given to the parent |
| max | the highest score of matching child doc is given to the parent |
| sum | the sum of scores of matching child doc is given to the parent |
| avg | the average of scores of matching child doc is given to the parent |
| none | default value, children scores are ignored |

__Avoid nested structure__, with Elastic Search !

We can recreate a SELECT of SELECT (from SQL) :
> Example of social network stories, made by some user (with id) and some other users can follow these user (field like `"following"=[1,3]`).
::: mermaid
graph LR;
S[Stories];
U[Users];
U --follows--> U;
S --made by--> U;
:::

Then we can retrieve the stories by the followed users with :
```js
GET /stories/_search
{
  "query": {
    "terms": {
      "user": {
         "index":"Users", // the index name
         "type":"_doc", // means it is an index
         "id":"1", // id of the user subscribed
         "path": "following" // get the list of subscription
      }
    }
  }
}
```
> Here the whole `user` object will be interpreted just as the `following` list of ids of the user with id=1.

## Formatting query result

* Prettify in cmd :
```js
curl -XGET "http://localhost:9200/recipes/_search?pretty" 
  -H 'Content-Type: application/json' 
  -d'{  "query": { // any query }}'
```

* Change to Yaml :
If we want rather a yaml output than a json.
```js
GET /stories/_search?format=yaml
{
  "query": {
    // any query
  }
}
```

* Limit the amount of data retrieved in `_source` :
```js
GET /stories/_search?format=yaml
{
  "_source": ...
  "query": {
    // any query
  }
}
```
The `_source` key can take many values, such has `false` if we want to retrieve only the ids of documents matching the query, or a key value of the source object to avoid getting all the document and only this specified key. Can be a list of keys such as `["title", "ingredients.*"]` to get only the name of the recipe and it's ingredients.

* Limit the number of documents to retrieve :

The default value of size is 10. This query below will still tell us how many docs are matching the query, but will only retrieve 2 complete docs with their values.
```js
GET /stories/_search?size=2
{
  "query": {
    "match_all": {}
  }
}
```

* Define an offset :

Can be done with `from` key (default value is 0). So using the `size` and `from` keys, we can retrieve only a defined amount of documents at a time (like 5 at a time, to show them in some web page, then the 5 next in another page).

$totalPages = ceil(\frac{totalHits}{pageSize})$ then the `from` key can take a value as $from = pageSize * (pageNumber - 1)$.

* Sorting :
```js
GET /stories/_search
{
  "_source": false,
  "query": {
    // any query
  },
  "sort" : [
    { "keyName" : "desc" } // name of the key to sort on (descending)
  ]
}
```
> Default value of a sorting is `asc`, in this case, no need to give an object in the `sort` list, just the name of the key is sufficient.

We can also sort on aggregated value. For exemple, if we have a `ratings` field which is a list of ratings :
```js
"sort" : [
  "ratings": {
    "order": "desc",
    "mode": "avg"
  }
]
```
That way we can get the average ratings, sorted by descending values.

* Highlight :
It is possible for ES to highlight (with `<em><\em>`) words in a text. This is useful, since the words of the query are going through the analyzer, and so are stemmed. Meaning even close words, that are good for the matching score, will also be highlighted.

```js
GET /indexName/_search
{
  "_source": false,
  "query": {
    "match": {
      "fieldName": // text we want to search and will be highlighted
    }
  },
  "highlight": {
    "pre_tags": ["<strong>"], // default = "<em>"
    "post_tags": ["</strong"], // default = "</em>"
    "fields": {
      "fieldName": {}
    }
  }
}
```
> With this query, the words matching in the field given will be highlighted in bold !

## Aggregations

* Metric Aggregation
We can do many aggregations in one query, and name them :
```js
GET /orders/_search
{
  "size": 0, // to get only the aggregations values
  "aggs": {
    "total_sales": {
      "sum": {
        "field": "total_amount"
      }
    },
    "avg_sales": {
      "avg": {
        "field": "total_amount"
      }
    },
    "min_sale": {
      "min": {
        "field": "total_amount"
      }
    },
    "max_sale": {
      "max": {
        "field": "total_amount"
      }
    }
  }
}
```
All the metric aggregations [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-metrics.html).
Note that there is a [`stat` aggregation](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-metrics-stats-aggregation.html) which returns count, min, max, avg and sum values at the same time.

* Nested Aggregation
With a term aggregation we can do statistics on one specific value :

```js
GET /orders/_search
{
  "size": 0,
  "aggs": {
    "status": {
      "terms": {
        "field": "status"
      },
      "aggs": {
        "stats_on_status": {
          "stats": {
            "field": "total_amount"
          }
        }
      }
    }
  }
}
```
With this query, we get the sales statistics on every order status found ! For the status `completed`, we have :
```yaml
    - key: "completed"
      doc_count: 204
      stats_on_status:
        count: 204
        min: 10.93
        max: 260.59
        avg: 113.54058823529411
        sum: 23162.28
```

* Buckets

When defining aggregations with `aggs` we can define _buckets_ (collection of documents) using a given rule. Can be ranges for example.
in the previous example, we created buckets according to the different status an order can get. Thanks to these buckets, we were able to do other aggregations on the elements of the different buckets.

Note that there's a pre-built aggregation `histogram` which will do on it's own the classification / bucketing (given the parameters).

