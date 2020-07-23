graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "categories"
      columns "name"
      types "string"
    pkey "id"
  ]
  node [
    id 1
    label "products"
      columns "product_name"
      columns "ingredients"
      columns "creator"
      columns "sugars_100g"
      columns "salt_100g"
      columns "fat_100g"
      columns "carbohydrates_100g"
      types "string"
      types "string"
      types "string"
      types "number"
      types "number"
      types "number"
      types "number"
    pkey "id"
  ]
  node [
    id 2
    label "brands"
      columns "name"
      types "string"
    pkey "id"
  ]
  node [
    id 3
    label "countries"
      columns "name"
      types "string"
    pkey "id"
  ]
  edge [
    source 1
    target 0
    key 0
    col1 "category"
    col2 "id"
    name "-"
  ]
  edge [
    source 2
    target 1
    key 0
    col1 "brand_id"
    col2 "product_id"
    name "products_brands"
  ]
  edge [
    source 3
    target 1
    key 0
    col1 "country_id"
    col2 "product_id"
    name "products_countries"
  ]
]
