"""Conexión JDBC a MySQL (misma estructura que la P0, que usaba SQL Server)."""


def obtener_url_conexion(host, puerto, base_datos):
    return (f"jdbc:mysql://{host}:{puerto}/{base_datos}"
            "?useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=UTC"
            "&rewriteBatchedStatements=true")


def obtener_propiedades_conexion(usuario, password):
    return {
        "driver": "com.mysql.cj.jdbc.Driver",
        "user": usuario,
        "password": password,
    }


def escribir_tabla(data_frame, nombre_tabla, url, propiedades, modo="overwrite"):
    (data_frame.write.option("truncate", "true")
     .jdbc(url=url, table=nombre_tabla, mode=modo, properties=propiedades))


def leer_tabla(spark_session, nombre_tabla, url, propiedades):
    return spark_session.read.jdbc(url=url, table=nombre_tabla, properties=propiedades)
