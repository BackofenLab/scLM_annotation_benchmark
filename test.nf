nextflow.enable.dsl=2

process download {

    script:
    """
    test.sh
    """
}

workflow {
    download()
}