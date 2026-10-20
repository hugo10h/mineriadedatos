'''
DECLARACIÓN USO DE IA

He utilizado la ayuda de Claude y ChatGPT para generar los códigos que 
conectan la practica con sql, ya que tenía mal instalado sql y me daba 
varios errores a la hora de cargar y ejecutar los códigos.
'''


def obtener_url_conexion(host, puerto, base_datos, instancia=None):
    if instancia:
        servidor = f"{host}\\{instancia}"
    else:
        servidor = f"{host}:{puerto}"

    return (
        f"jdbc:sqlserver://{servidor};"
        f"databaseName={base_datos};"
        "encrypt=false;trustServerCertificate=true"
    )


def obtener_propiedades_conexion(usuario, password):
    return {
        "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver",
        "user": usuario,
        "password": password,
    }


def escribir_tabla(data_frame, nombre_tabla, url, propiedades, modo="overwrite"):
    data_frame.write.jdbc(url=url, table=nombre_tabla, mode=modo, properties=propiedades)


def leer_tabla(spark_session, nombre_tabla, url, propiedades):
    return spark_session.read.jdbc(url=url, table=nombre_tabla, properties=propiedades)